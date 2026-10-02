

"""
Pool class for Parallel Tempering in symbolic regression.

Each pool maintains a population of programs at a specific temperature.
Cold pools (high beta/inverse temp) favor exploitation of high-scoring programs.
Hot pools (low beta) favor exploration of diverse programs.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Any, Mapping, Dict
from collections import defaultdict, deque
import copy
import numpy as np

from llmsr import code_manipulation
from llmsr.pt_math_utils import softmax_np, softmax_probs

# Type alias for scores per test
ScoresPerTest = Mapping[Any, float]


def _reduce_score(scores_per_test: ScoresPerTest) -> float:
    """Reduce per-test scores to a single score (average)."""
    test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
    return sum(test_scores) / len(test_scores) if test_scores else 0.0


class PTPool:
    """
    A single-temperature population (pool) for parallel tempering.
    
    Maintains a collection of programs with their scores, supporting:
    - Temperature-weighted sampling for mating pool creation
    - Population update with new offspring
    - Diversity tracking and optional diversity-aware selection
    
    Args:
        temperature: Inverse temperature (beta). Higher = more exploitation.
        population_size: Maximum number of programs to maintain.
        island_id: ID of the parent island (for logging).
        pool_id: ID of this pool within the island.
    """
    
    def __init__(
        self,
        temperature: float,
        population_size: int,
        island_id: int = 0,
        pool_id: int = 0,
    ):
        self.temperature = float(temperature)  # beta (inverse temperature)
        self.population_size = int(population_size)
        self.island_id = island_id
        self.pool_id = pool_id
        
        # Program storage
        self.programs: List[code_manipulation.Function] = []
        self.scores: List[float] = []
        self.scores_per_test: List[ScoresPerTest] = []
        
        # OOD (out-of-distribution) score storage for diversity analysis
        self.ood_scores: List[Optional[float]] = []
        self.ood_scores_per_test: List[Optional[ScoresPerTest]] = []
        
        # Tracking statistics
        self.swap_stats = defaultdict(lambda: {"proposals": 0, "accepts": 0})
        self.swap_hist = defaultdict(lambda: deque(maxlen=200))
        self._num_programs_added: int = 0
        
    def __len__(self) -> int:
        return len(self.programs)
    
    @property
    def is_empty(self) -> bool:
        return len(self.programs) == 0
    
    def register_program(
        self,
        program: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
        ood_scores_per_test: Optional[ScoresPerTest] = None,
    ) -> None:
        """
        Add a program to the pool.
        
        If pool is at capacity, replace the worst program if the new one is better.
        Deduplicates by program body.
        
        Args:
            program: The program function to register
            scores_per_test: Per-test scores on training data
            ood_scores_per_test: Optional per-test scores on OOD (out-of-distribution) data
        """
        score = _reduce_score(scores_per_test)
        ood_score = _reduce_score(ood_scores_per_test) if ood_scores_per_test else None
        
        # Check for duplicates by body
        new_body = program.body.strip()
        for existing_prog in self.programs:
            if existing_prog.body.strip() == new_body:
                return  # Skip duplicates
        
        if len(self.programs) < self.population_size:
            # Pool not full, just add
            self.programs.append(copy.deepcopy(program))
            self.scores.append(score)
            self.scores_per_test.append(dict(scores_per_test))
            self.ood_scores.append(ood_score)
            self.ood_scores_per_test.append(dict(ood_scores_per_test) if ood_scores_per_test else None)
        else:
            # Pool full, replace worst if new is better
            min_idx = int(np.argmin(self.scores))
            if score > self.scores[min_idx]:
                self.programs[min_idx] = copy.deepcopy(program)
                self.scores[min_idx] = score
                self.scores_per_test[min_idx] = dict(scores_per_test)
                self.ood_scores[min_idx] = ood_score
                self.ood_scores_per_test[min_idx] = dict(ood_scores_per_test) if ood_scores_per_test else None
        
        self._num_programs_added += 1
    
    def get_program_bodies(self) -> List[str]:
        """Return list of program bodies for deduplication."""
        return [p.body.strip() for p in self.programs]
    
    def sample_program(
        self,
        gamma: float = 1.0,
        eps_mix: float = 0.0,
    ) -> Tuple[code_manipulation.Function, float, int]:
        """
        Sample a program using temperature-weighted probabilities.
        
        Args:
            gamma: Power to raise probabilities to (gamma < 1 flattens distribution)
            eps_mix: Uniform mixing coefficient for exploration
            
        Returns:
            Tuple of (program, score, index)
        """
        if self.is_empty:
            raise ValueError("Cannot sample from empty pool")
        
        N = len(self.programs)
        
        # Base probability using beta-weighted softmax
        probs = softmax_probs(self.scores, self.temperature)
        
        # Apply gamma power (flattening if gamma < 1)
        if gamma != 1.0 and gamma > 0:
            probs = np.power(probs, gamma)
            probs = probs / probs.sum()
        
        # Mix with uniform for exploration
        if eps_mix > 0.0:
            probs = (1.0 - eps_mix) * probs + eps_mix * (1.0 / N)
            probs = probs / probs.sum()
        
        idx = int(np.random.choice(N, p=probs))
        return copy.deepcopy(self.programs[idx]), self.scores[idx], idx
    
    def make_mating_pool(
        self,
        num_samples: int,
        gamma: float = 1.0,
        eps_mix: float = 0.0,
    ) -> List[Tuple[code_manipulation.Function, float]]:
        """
        Create a mating pool by sampling programs with replacement.
        
        Args:
            num_samples: Number of programs to sample
            gamma: Power for probability flattening
            eps_mix: Uniform mixing coefficient
            
        Returns:
            List of (program, score) tuples
        """
        if self.is_empty:
            return []
        
        mating_pool = []
        for _ in range(num_samples):
            prog, score, _ = self.sample_program(gamma=gamma, eps_mix=eps_mix)
            mating_pool.append((prog, score))
        
        return mating_pool
    
    def get_best_programs(self, k: int) -> List[Tuple[code_manipulation.Function, float, ScoresPerTest]]:
        """Return the k best programs by score."""
        if self.is_empty:
            return []
        
        indices = np.argsort(self.scores)[::-1][:k]
        return [
            (copy.deepcopy(self.programs[i]), self.scores[i], self.scores_per_test[i])
            for i in indices
        ]
    
    def get_best_score(self) -> float:
        """Return the best score in the pool."""
        return max(self.scores) if self.scores else float('-inf')
    
    def get_mean_score(self) -> float:
        """Return the mean score in the pool."""
        return np.mean(self.scores) if self.scores else 0.0
    
    def compute_diversity(self) -> float:
        """
        Compute a simple diversity metric based on unique program structures.
        Returns ratio of unique bodies to total programs.
        """
        if len(self.programs) <= 1:
            return 1.0
        
        unique_bodies = set(p.body.strip() for p in self.programs)
        return len(unique_bodies) / len(self.programs)
    
    def replace_at_index(
        self,
        idx: int,
        program: code_manipulation.Function,
        score: float,
        scores_per_test: ScoresPerTest,
        ood_score: Optional[float] = None,
        ood_scores_per_test: Optional[ScoresPerTest] = None,
    ) -> None:
        """Replace program at specific index (used during swaps)."""
        if 0 <= idx < len(self.programs):
            self.programs[idx] = copy.deepcopy(program)
            self.scores[idx] = score
            self.scores_per_test[idx] = dict(scores_per_test)
            
            # Handle OOD scores
            if idx < len(self.ood_scores):
                self.ood_scores[idx] = ood_score
                self.ood_scores_per_test[idx] = dict(ood_scores_per_test) if ood_scores_per_test else None
    
    def prune_for_diversity(
        self,
        keep_size: int,
        elites: int = 5,
    ) -> None:
        """
        Reduce pool size while preserving diversity.
        Keeps top `elites` programs and fills rest with diverse selection.
        """
        if len(self.programs) <= keep_size:
            return
        
        # Sort by score descending
        indices = np.argsort(self.scores)[::-1]
        
        # Keep elites
        kept_indices = list(indices[:elites])
        kept_bodies = set(self.programs[i].body.strip() for i in kept_indices)
        
        # Fill remaining slots with diverse programs
        for idx in indices[elites:]:
            if len(kept_indices) >= keep_size:
                break
            body = self.programs[idx].body.strip()
            if body not in kept_bodies:
                kept_indices.append(idx)
                kept_bodies.add(body)
        
        # If still need more, add remaining high-scoring ones
        for idx in indices[elites:]:
            if len(kept_indices) >= keep_size:
                break
            if idx not in kept_indices:
                kept_indices.append(idx)
        
        # Rebuild pool
        new_programs = [self.programs[i] for i in kept_indices]
        new_scores = [self.scores[i] for i in kept_indices]
        new_scores_per_test = [self.scores_per_test[i] for i in kept_indices]
        
        # Also preserve OOD scores
        new_ood_scores = []
        new_ood_scores_per_test = []
        for i in kept_indices:
            if i < len(self.ood_scores):
                new_ood_scores.append(self.ood_scores[i])
                new_ood_scores_per_test.append(self.ood_scores_per_test[i])
            else:
                new_ood_scores.append(None)
                new_ood_scores_per_test.append(None)
        
        self.programs = new_programs
        self.scores = new_scores
        self.scores_per_test = new_scores_per_test
        self.ood_scores = new_ood_scores
        self.ood_scores_per_test = new_ood_scores_per_test

    def get_all_programs_for_snapshot(self) -> List[dict]:
        """Get all programs in this pool for snapshot saving.
        
        Returns a list of dictionaries with program info including OOD scores.
        """
        programs_info = []
        for i, program in enumerate(self.programs):
            prog_info = {
                "program_body": program.body,
                "program_name": program.name,
                "program_args": program.args,
                "program_docstring": program.docstring,
                "program_length": len(program.body),
                "score": self.scores[i],
                "scores_per_test": dict(self.scores_per_test[i]) if self.scores_per_test[i] else {},
            }
            
            # Add OOD scores if available
            if i < len(self.ood_scores):
                prog_info["ood_score"] = self.ood_scores[i]
                prog_info["ood_scores_per_test"] = (
                    dict(self.ood_scores_per_test[i]) 
                    if self.ood_scores_per_test[i] else {}
                )
            else:
                prog_info["ood_score"] = None
                prog_info["ood_scores_per_test"] = {}
            
            programs_info.append(prog_info)
        return programs_info


