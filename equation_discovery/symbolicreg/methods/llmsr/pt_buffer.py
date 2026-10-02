

"""
PT-aware Experience Buffer for symbolic regression.

Each island contains multiple parallel-tempering pools at different
temperatures, with periodic migration between pools.
"""
from __future__ import annotations

import dataclasses
import time
import copy
import os
import json
from collections import deque, defaultdict
from typing import Any, Tuple, Mapping, List, Optional, Dict

from absl import logging
import numpy as np
import scipy

from llmsr import code_manipulation
from llmsr import config as config_lib
from llmsr.pt_pool import PTPool, _reduce_score
from llmsr.pt_migration import pt_migration_softmax_random_inplace, compute_swap_rates
from llmsr.pt_records import SwapRecord, export_swap_records_to_csv, export_swap_records_to_jsonl


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
    """Represents test scores as a canonical signature."""
    return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))


@dataclasses.dataclass(frozen=True)
class PTPrompt:
    """A prompt produced by the PT Experience Buffer.

    Args:
        code: The prompt, ending with the header of the function to be completed.
        version_generated: The function to be completed is `_v{version_generated}`.
        island_id: Identifier of the island that produced the samples.
        pool_id: Identifier of the pool within the island.
    """
    code: str
    version_generated: int
    island_id: int
    pool_id: int


class PTExperienceBuffer:
    """
    Experience buffer with parallel tempering pools.
    
    Each island contains multiple pools at different temperatures:
    - Cold pool (high beta): favors exploitation of high-scoring programs
    - Hot pool (low beta): favors exploration of diverse programs
    
    Periodic migration swaps programs between pools based on PT criterion.
    """

    def __init__(
        self,
        config: config_lib.PTExperienceBufferConfig,
        template: code_manipulation.Program,
        function_to_evolve: str,
        log_dir: Optional[str] = None,
    ) -> None:
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._log_dir = log_dir
        
        # PT temperatures (inverse temperatures / beta values)
        # Higher beta = colder = more exploitation
        self._temperatures = config.pool_temperatures
        self._num_pools = len(self._temperatures)
        
        # Initialize islands, each containing multiple PT pools
        self._islands: List[PTIsland] = []
        for island_id in range(config.num_islands):
            island = PTIsland(
                island_id=island_id,
                template=template,
                function_to_evolve=function_to_evolve,
                temperatures=self._temperatures,
                population_size=config.population_size,
                functions_per_prompt=config.functions_per_prompt,
                cluster_sampling_temperature_init=config.cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=config.cluster_sampling_temperature_period,
            )
            self._islands.append(island)
        
        # Global best tracking per island
        self._best_score_per_island: List[float] = [-float('inf')] * config.num_islands
        self._best_program_per_island: List[Optional[code_manipulation.Function]] = [None] * config.num_islands
        self._best_scores_per_test_per_island: List[Optional[ScoresPerTest]] = [None] * config.num_islands
        
        # Migration tracking
        self._migration_step = 0
        self._global_step = 0
        self._swap_log: List[SwapRecord] = []
        self._swap_stats_global = {"proposals": 0, "accepts": 0}
        self._swap_hist_global = deque(maxlen=200)
        
        # Adaptive Boltzmann constant (ξ) tracking
        # Initialize xi and swap rate window (Γ) as per Algorithm 1
        self._xi = config.xi_init  # Current Boltzmann constant (ξ)
        self._swap_rate_window: List[float] = []  # Γ in the algorithm
        self._xi_history: List[dict] = []  # Track xi adaptation history
        
        # Timing
        self._last_reset_time = time.time()
        self._last_migration_time = time.time()
        
        # Logging setup
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._csv_path = os.path.join(log_dir, 'pt_swap_log.csv')
            self._jsonl_path = os.path.join(log_dir, 'pt_swap_log.jsonl')
        else:
            self._csv_path = None
            self._jsonl_path = None
        
        # Buffer snapshot saving configuration for diversity analysis
        self._snapshot_interval = config.snapshot_interval
        self._snapshot_count = 0
        
        # Create snapshot directory if needed
        if self._log_dir and self._snapshot_interval > 0:
            self._snapshot_dir = os.path.join(self._log_dir, 'buffer_snapshots')
            os.makedirs(self._snapshot_dir, exist_ok=True)
        else:
            self._snapshot_dir = None

    def get_prompt(self) -> PTPrompt:
        """Returns a prompt from a randomly chosen island and pool."""
        island_id = np.random.randint(len(self._islands))
        island = self._islands[island_id]
        
        # Choose pool based on configured strategy
        if self._config.pool_selection_strategy == "uniform":
            pool_id = np.random.randint(self._num_pools)
        elif self._config.pool_selection_strategy == "cold_biased":
            # Bias towards cold pool (index 0) for exploitation
            weights = [2.0] + [1.0] * (self._num_pools - 1)
            weights = np.array(weights) / sum(weights)
            pool_id = int(np.random.choice(self._num_pools, p=weights))
        else:
            pool_id = 0  # Default to cold pool
        
        code, version_generated = island.get_prompt(pool_id)
        return PTPrompt(code, version_generated, island_id, pool_id)

    def register_program(
        self,
        program: code_manipulation.Function,
        island_id: Optional[int],
        scores_per_test: ScoresPerTest,
        pool_id: int = 0,
        ood_scores_per_test: Optional[ScoresPerTest] = None,
        **kwargs
    ) -> None:
        """Register a new program in the appropriate pool.
        
        Args:
            program: The program function to register
            island_id: Target island ID (None for all islands)
            scores_per_test: Per-test scores on training data
            pool_id: Target pool ID within the island
            ood_scores_per_test: Optional per-test scores on OOD data
            **kwargs: Additional arguments including profiler
        """
        self._global_step += 1
        
        if island_id is None:
            # Register in all islands (initial seeding)
            for iid in range(len(self._islands)):
                self._register_program_in_island(
                    program, iid, scores_per_test, pool_id, 
                    ood_scores_per_test=ood_scores_per_test, **kwargs
                )
        else:
            self._register_program_in_island(
                program, island_id, scores_per_test, pool_id,
                ood_scores_per_test=ood_scores_per_test, **kwargs
            )
        
        # Periodic buffer snapshot saving for diversity analysis
        if self._snapshot_dir and self._snapshot_interval > 0:
            if self._global_step % self._snapshot_interval == 0:
                self._save_buffer_snapshot()
        
        # Check if it's time for migration
        if self._should_perform_migration():
            self._perform_migration()
        
        # Check island reset
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self._reset_islands()

    def _register_program_in_island(
        self,
        program: code_manipulation.Function,
        island_id: int,
        scores_per_test: ScoresPerTest,
        pool_id: int,
        ood_scores_per_test: Optional[ScoresPerTest] = None,
        **kwargs
    ) -> None:
        """Register program in a specific island's pool."""
        island = self._islands[island_id]
        island.register_program(program, scores_per_test, pool_id, ood_scores_per_test=ood_scores_per_test)
        
        # Update best tracking
        score = _reduce_score(scores_per_test)
        ood_score = _reduce_score(ood_scores_per_test) if ood_scores_per_test else None
        
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)
        
        # Profiler logging with OOD score and pool info
        profiler = kwargs.get('profiler', None)
        if profiler:
            program.score = score
            program.ood_score = ood_score  # Add OOD score
            program.pool_id = pool_id  # Add pool ID
            program.island_id = island_id  # Add island ID
            program.global_sample_nums = kwargs.get('global_sample_nums', None)
            program.sample_time = kwargs.get('sample_time', None)
            program.evaluate_time = kwargs.get('evaluate_time', None)
            profiler.register_function(program)

    def _should_perform_migration(self) -> bool:
        """Check if migration should be performed."""
        # Migration based on number of programs registered
        if self._global_step % self._config.migration_interval == 0:
            return True
        return False

    def _perform_migration(self) -> None:
        """Perform PT migration between pools in all islands.
        
        Implements Algorithm 1 lines 12-20: PT swap with adaptive Boltzmann constant.
        After each migration window, adjusts ξ to keep swap rate in target range.
        """
        migration_proposals = 0
        migration_accepts = 0
        
        for island in self._islands:
            records = island.perform_migration(
                migration_size=self._config.migration_size,
                step=self._global_step,
                xi=self._xi,  # Pass current Boltzmann constant
            )
            
            # Aggregate statistics
            for rec in records:
                if rec.accepted:
                    self._swap_stats_global["accepts"] += 1
                    self._swap_hist_global.append(1)
                    migration_accepts += 1
                else:
                    self._swap_hist_global.append(0)
                self._swap_stats_global["proposals"] += 1
                migration_proposals += 1
            
            self._swap_log.extend(records)
        
        self._migration_step += 1
        self._last_migration_time = time.time()
        
        # Algorithm 1, line 14: Append γ (swap rate for this migration) to Γ
        if migration_proposals > 0:
            gamma = migration_accepts / migration_proposals
            self._swap_rate_window.append(gamma)
        
        # Algorithm 1, lines 15-20: Adapt Boltzmann constant when window is full
        self._adapt_boltzmann_constant()
        
        # Export logs periodically
        if self._migration_step % self._config.log_every_migrations == 0:
            self._export_swap_logs()
    
    def _adapt_boltzmann_constant(self) -> None:
        """Adapt the Boltzmann constant (ξ) based on swap rate.
        
        Implements Algorithm 1 lines 15-20:
        - If window Γ is full (length == L):
            - If mean(Γ) >= τ + ε/2: multiply ξ by the configured decay factor
            - If mean(Γ) <= τ - ε/2: multiply ξ by the configured growth factor
            - Reset Γ to empty
        
        Scaling is adjusted when the observed rate is outside the target range.
        """
        L = self._config.swap_rate_window_size
        tau = self._config.swap_rate_target
        epsilon = self._config.swap_rate_tolerance
        
        # Check if window is full (Algorithm 1, line 15)
        if len(self._swap_rate_window) < L:
            return
        
        # Compute mean swap rate over the window
        mean_gamma = sum(self._swap_rate_window) / len(self._swap_rate_window)
        old_xi = self._xi
        action = "none"
        
        # Algorithm 1, lines 16-17: If swap rate too high, reduce ξ
        if mean_gamma >= tau + epsilon / 2:
            self._xi = self._xi * self._config.xi_growth
            action = "growth"
        # Algorithm 1, lines 18-19: If swap rate too low, increase ξ
        elif mean_gamma <= tau - epsilon / 2:
            self._xi = self._xi * self._config.xi_decay
            action = "decay"
        
        # Clamp ξ to valid range
        self._xi = max(self._config.xi_min, min(self._config.xi_max, self._xi))
        
        # Log the adaptation event
        adaptation_record = {
            "step": self._global_step,
            "migration_step": self._migration_step,
            "mean_swap_rate": mean_gamma,
            "target": tau,
            "tolerance": epsilon,
            "old_xi": old_xi,
            "new_xi": self._xi,
            "action": action,
            "window_size": len(self._swap_rate_window),
        }
        self._xi_history.append(adaptation_record)
        
        # Print adaptation info
        if action != "none":
            logging.info(
                f"PT ξ adaptation (step {self._global_step}): "
                f"mean_swap_rate={mean_gamma:.3f}, target=[{tau - epsilon/2:.3f}, {tau + epsilon/2:.3f}], "
                f"ξ: {old_xi:.4f} -> {self._xi:.4f} ({action})"
            )
        
        # Algorithm 1, line 20: Reset window Γ
        self._swap_rate_window = []
    
    def get_pool_statistics(self) -> Dict[int, Dict[str, float]]:
        """Get statistics for all pools across all islands.
        
        Returns:
            Dictionary mapping pool_id to stats dict with keys:
            'best_score', 'mean_score', 'diversity', 'size'
        """
        pool_stats = {}
        
        for island in self._islands:
            for pool in island._pools:
                pool_id = pool.pool_id
                if pool_id not in pool_stats:
                    pool_stats[pool_id] = {
                        'best_score': float('-inf'),
                        'mean_score': 0.0,
                        'diversity': 0.0,
                        'size': 0,
                        'total_scores': [],
                    }
                
                if not pool.is_empty:
                    pool_stats[pool_id]['best_score'] = max(
                        pool_stats[pool_id]['best_score'],
                        pool.get_best_score()
                    )
                    pool_stats[pool_id]['total_scores'].extend(pool.scores)
                    pool_stats[pool_id]['size'] += len(pool)
                    # Use max diversity across islands for each pool
                    pool_stats[pool_id]['diversity'] = max(
                        pool_stats[pool_id]['diversity'],
                        pool.compute_diversity()
                    )
        
        # Compute mean scores
        for pool_id, stats in pool_stats.items():
            if stats['total_scores']:
                stats['mean_score'] = sum(stats['total_scores']) / len(stats['total_scores'])
            del stats['total_scores']  # Remove temporary field
        
        return pool_stats
    
    def update_profiler_pt_metrics(self, profiler) -> None:
        """Update the profiler with PT-specific metrics.
        
        Call this periodically (e.g., after migration) to log PT metrics to tensorboard.
        
        Args:
            profiler: The Profiler instance to update
        """
        if profiler is None:
            return
        
        # Get swap rates
        cum_rate, roll_rate = self.get_global_swap_rate()
        
        # Get pool statistics
        pool_stats = self.get_pool_statistics()
        
        # Update profiler
        if hasattr(profiler, 'update_pt_metrics'):
            profiler.update_pt_metrics(
                swap_rate_cumulative=cum_rate,
                swap_rate_rolling=roll_rate,
                xi=self._xi,
                pool_stats=pool_stats,
            )
        
        # Log score distribution
        if hasattr(profiler, 'log_score_distribution'):
            profiler.log_score_distribution()

    def _export_swap_logs(self) -> None:
        """Export swap logs and xi adaptation history to files."""
        if self._swap_log:
            if self._csv_path:
                export_swap_records_to_csv(self._swap_log, self._csv_path, append=True)
            if self._jsonl_path:
                export_swap_records_to_jsonl(self._swap_log, self._jsonl_path, append=True)
            self._swap_log = []
        
        # Export xi adaptation history
        if self._xi_history and self._log_dir:
            xi_path = os.path.join(self._log_dir, 'xi_adaptation_log.jsonl')
            with open(xi_path, 'w', encoding='utf-8') as f:
                for record in self._xi_history:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Also export current stats
            stats_path = os.path.join(self._log_dir, 'swap_rate_stats.json')
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self.get_swap_rate_stats(), f, indent=2)
    
    def _save_buffer_snapshot(self) -> None:
        """Save buffer snapshot for diversity analysis.
        
        Saves all programs in the buffer with their pool_id, island_id, 
        program_length, and score.
        """
        if not self._snapshot_dir:
            return
        
        self._snapshot_count += 1
        snapshot_data = {
            "step": self._global_step,
            "timestamp": time.time(),
            "snapshot_count": self._snapshot_count,
            "num_islands": len(self._islands),
            "num_pools": self._num_pools,
            "temperatures": list(self._temperatures),
            "current_xi": self._xi,
            "programs": [],
        }
        
        # Collect programs from all islands and pools
        for island in self._islands:
            island_programs = island.get_all_programs_for_snapshot()
            for prog_info in island_programs:
                snapshot_data["programs"].append(prog_info)
        
        # Add summary statistics
        snapshot_data["total_programs"] = len(snapshot_data["programs"])
        snapshot_data["programs_per_island"] = {
            i: sum(1 for p in snapshot_data["programs"] if p["island_id"] == i)
            for i in range(len(self._islands))
        }
        snapshot_data["programs_per_pool"] = {
            p: sum(1 for prog in snapshot_data["programs"] if prog["pool_id"] == p)
            for p in range(self._num_pools)
        }
        
        # Save to file
        snapshot_path = os.path.join(
            self._snapshot_dir,
            f"snapshot_step{self._global_step:06d}.json"
        )
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=2, ensure_ascii=False)
        
        logging.info(
            f"Saved PT buffer snapshot at step {self._global_step} with "
            f"{snapshot_data['total_programs']} programs to {snapshot_path}"
        )

    def _reset_islands(self) -> None:
        """Reset the weaker half of islands."""
        # Sort by best score with noise to break ties
        indices_sorted = np.argsort(
            self._best_score_per_island + 
            np.random.randn(len(self._best_score_per_island)) * 1e-6
        )
        
        num_to_reset = self._config.num_islands // 2
        reset_ids = indices_sorted[:num_to_reset]
        keep_ids = indices_sorted[num_to_reset:]
        
        for island_id in reset_ids:
            # Create fresh island
            self._islands[island_id] = PTIsland(
                island_id=island_id,
                template=self._template,
                function_to_evolve=self._function_to_evolve,
                temperatures=self._temperatures,
                population_size=self._config.population_size,
                functions_per_prompt=self._config.functions_per_prompt,
                cluster_sampling_temperature_init=self._config.cluster_sampling_temperature_init,
                cluster_sampling_temperature_period=self._config.cluster_sampling_temperature_period,
            )
            self._best_score_per_island[island_id] = -float('inf')
            
            # Seed with best from a kept island
            founder_id = np.random.choice(keep_ids)
            founder = self._best_program_per_island[founder_id]
            founder_scores = self._best_scores_per_test_per_island[founder_id]
            if founder and founder_scores:
                self._register_program_in_island(founder, island_id, founder_scores, pool_id=0)

    def get_global_swap_rate(self) -> Tuple[float, float]:
        """Return global cumulative and rolling swap rates."""
        total = self._swap_stats_global["proposals"]
        acc = self._swap_stats_global["accepts"]
        cum = acc / total if total > 0 else float('nan')
        roll = sum(self._swap_hist_global) / len(self._swap_hist_global) if self._swap_hist_global else float('nan')
        return cum, roll
    
    @property
    def xi(self) -> float:
        """Return current Boltzmann constant (ξ)."""
        return self._xi
    
    @property
    def xi_history(self) -> List[dict]:
        """Return history of ξ adaptations."""
        return self._xi_history
    
    def get_swap_rate_stats(self) -> dict:
        """Return comprehensive swap rate and ξ statistics."""
        cum_rate, roll_rate = self.get_global_swap_rate()
        window_mean = (
            sum(self._swap_rate_window) / len(self._swap_rate_window) 
            if self._swap_rate_window else float('nan')
        )
        return {
            "cumulative_swap_rate": cum_rate,
            "rolling_swap_rate": roll_rate,
            "current_window_mean": window_mean,
            "current_window_size": len(self._swap_rate_window),
            "target_window_size": self._config.swap_rate_window_size,
            "current_xi": self._xi,
            "xi_init": self._config.xi_init,
            "target_swap_rate": self._config.swap_rate_target,
            "tolerance": self._config.swap_rate_tolerance,
            "num_adaptations": len(self._xi_history),
            "total_proposals": self._swap_stats_global["proposals"],
            "total_accepts": self._swap_stats_global["accepts"],
        }


class PTIsland:
    """
    An island containing multiple PT pools at different temperatures.
    
    Programs flow between pools via migration, encouraging diversity
    while maintaining high-quality solutions.
    """

    def __init__(
        self,
        island_id: int,
        template: code_manipulation.Program,
        function_to_evolve: str,
        temperatures: List[float],
        population_size: int,
        functions_per_prompt: int,
        cluster_sampling_temperature_init: float,
        cluster_sampling_temperature_period: int,
    ):
        self.island_id = island_id
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._functions_per_prompt = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period
        
        # Create pools at different temperatures
        self._pools: List[PTPool] = []
        for pool_id, temp in enumerate(temperatures):
            pool = PTPool(
                temperature=temp,
                population_size=population_size,
                island_id=island_id,
                pool_id=pool_id,
            )
            self._pools.append(pool)
        
        # Migration tracking per pool pair
        self._swap_stats = defaultdict(lambda: {"proposals": 0, "accepts": 0})
        self._swap_hist = defaultdict(lambda: deque(maxlen=200))
        
        self._num_programs = 0

    def register_program(
        self,
        program: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
        pool_id: int = 0,
        ood_scores_per_test: Optional[ScoresPerTest] = None,
    ) -> None:
        """Register a program in the specified pool.
        
        Args:
            program: The program function to register
            scores_per_test: Per-test scores on training data
            pool_id: Target pool ID
            ood_scores_per_test: Optional per-test scores on OOD data
        """
        if 0 <= pool_id < len(self._pools):
            self._pools[pool_id].register_program(
                program, scores_per_test, ood_scores_per_test=ood_scores_per_test
            )
        self._num_programs += 1

    def get_prompt(self, pool_id: int = 0) -> Tuple[str, int]:
        """Get a prompt by sampling from the specified pool."""
        pool = self._pools[pool_id]
        
        if pool.is_empty:
            # Fallback to initial template if pool is empty
            return self._generate_initial_prompt(), 1
        
        # Sample programs for prompt
        num_to_sample = min(self._functions_per_prompt, len(pool))
        samples = pool.make_mating_pool(num_to_sample, gamma=0.8, eps_mix=0.05)
        
        implementations = [prog for prog, _ in samples]
        scores = [score for _, score in samples]
        
        # Sort by score ascending (worst to best)
        sorted_pairs = sorted(zip(scores, implementations), key=lambda x: x[0])
        sorted_implementations = [impl for _, impl in sorted_pairs]
        
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated

    def _generate_initial_prompt(self) -> str:
        """Generate initial prompt when pools are empty."""
        func = self._template.get_function(self._function_to_evolve)
        header = dataclasses.replace(
            func,
            name=f'{self._function_to_evolve}_v1',
            body='',
            docstring='Initial version of the function.',
        )
        prompt = dataclasses.replace(self._template, functions=[header])
        return str(prompt)

    def _generate_prompt(
        self,
        implementations: List[code_manipulation.Function]
    ) -> str:
        """Generate prompt from a list of implementations."""
        implementations = copy.deepcopy(implementations)
        
        versioned_functions: List[code_manipulation.Function] = []
        for i, implementation in enumerate(implementations):
            new_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_name
            if i >= 1:
                implementation.docstring = f'Improved version of `{self._function_to_evolve}_v{i - 1}`.'
            
            # Handle recursive calls
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_name
            )
            versioned_functions.append(code_manipulation.text_to_function(implementation))
        
        # Create header for new function
        next_version = len(implementations)
        new_name = f'{self._function_to_evolve}_v{next_version}'
        header = dataclasses.replace(
            implementations[-1],
            name=new_name,
            body='',
            docstring=f'Improved version of `{self._function_to_evolve}_v{next_version - 1}`.',
        )
        versioned_functions.append(header)
        
        prompt = dataclasses.replace(self._template, functions=versioned_functions)
        return str(prompt)

    def perform_migration(
        self,
        migration_size: int,
        step: Optional[int] = None,
        xi: float = 2.5,
    ) -> List[SwapRecord]:
        """
        Perform PT migration between pools.
        
        Attempt swaps between each adjacent pair of pools.
        
        Args:
            migration_size: Number of swap attempts per pool pair
            step: Current global step (for logging)
            xi: Boltzmann constant (ξ) for computing acceptance probability.
                This is adaptively adjusted to maintain target swap rate.
        """
        all_records = []
        
        if len(self._pools) < 2:
            return all_records
        
        # Perform migration between adjacent temperature pools
        for i in range(len(self._pools) - 1):
            pool_a = self._pools[i]      # Colder
            pool_b = self._pools[i + 1]  # Hotter
            
            if pool_a.is_empty or pool_b.is_empty:
                continue
            
            proposals, acc_a, acc_b, records = pt_migration_softmax_random_inplace(
                pool_a=pool_a,
                pool_b=pool_b,
                migration_size=migration_size,
                island_id=self.island_id,
                pool_a_idx=i,
                pool_b_idx=i + 1,
                step=step,
                xi=xi,  # Pass Boltzmann constant
            )
            
            # Update statistics
            key = (i, i + 1)
            self._swap_stats[key]["proposals"] += proposals
            self._swap_stats[key]["accepts"] += acc_a
            self._swap_hist[key].extend([1] * acc_a)
            self._swap_hist[key].extend([0] * (proposals - acc_a))
            
            all_records.extend(records)
            
            cum_rate, roll_rate = compute_swap_rates(self._swap_stats, self._swap_hist, key)
            print(
                f"  Island {self.island_id}: Migration pool {i} (β={pool_a.temperature:.3g}) "
                f"<-> pool {i+1} (β={pool_b.temperature:.3g}); "
                f"proposals={proposals}, accepted={acc_a}, cum_rate={cum_rate:.3f}, ξ={xi:.4f}"
            )
        
        return all_records
    
    def get_all_programs_for_snapshot(self) -> List[dict]:
        """Get all programs in this island for snapshot saving.
        
        Returns a list of dictionaries, each containing:
        - program_body: The code body of the program
        - program_name: Function name
        - program_args: Function arguments
        - program_docstring: Function docstring
        - program_length: Length of the program body
        - score: Program score
        - scores_per_test: Per-test scores
        - island_id: ID of this island
        - pool_id: ID of the pool containing this program
        - pool_temperature: Temperature (beta) of the pool
        """
        all_programs = []
        
        for pool in self._pools:
            pool_programs = pool.get_all_programs_for_snapshot()
            for prog_info in pool_programs:
                # Add island and pool information
                prog_info["island_id"] = self.island_id
                prog_info["pool_id"] = pool.pool_id
                prog_info["pool_temperature"] = pool.temperature
                all_programs.append(prog_info)
        
        return all_programs
