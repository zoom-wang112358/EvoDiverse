# Record search metrics with TensorBoard.

from __future__ import annotations

import os.path
from typing import List, Dict, Optional, Any
import logging
import json
import numpy as np
from llmsr import code_manipulation
from tensorboardX import SummaryWriter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -np.inf
        self._cur_best_program_str = None
        self._cur_best_program_ood_score = None  # OOD score of best program
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []
        
        # Track OOD scores over iterations
        self._each_sample_best_ood_score = []
        
        # PT-specific tracking
        self._pt_swap_rate_cumulative = []
        self._pt_swap_rate_rolling = []
        self._pt_xi_values = []
        self._pt_pool_best_scores: Dict[int, List[float]] = {}
        self._pt_pool_diversity: Dict[int, List[float]] = {}
        
        # Score distribution tracking
        self._score_history: List[float] = []
        self._ood_score_history: List[float] = []


    def _write_tensorboard(self, function: Optional[code_manipulation.Function] = None):
        if not self._log_dir:
            return

        # Best score tracking (In-Distribution)
        self._writer.add_scalar(
            'Best/ID_Score',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        
        # OOD score of the best program
        if self._cur_best_program_ood_score is not None:
            self._writer.add_scalar(
                'Best/OOD_Score',
                self._cur_best_program_ood_score,
                global_step=self._num_samples
            )
        
        # Track current sample's scores (not just best)
        if function is not None and function.score is not None:
            self._writer.add_scalar(
                'Current/ID_Score',
                function.score,
                global_step=self._num_samples
            )
            if function.ood_score is not None:
                self._writer.add_scalar(
                    'Current/OOD_Score',
                    function.ood_score,
                    global_step=self._num_samples
                )
            
            # Track pool information if available
            if function.pool_id is not None:
                self._writer.add_scalar(
                    'Current/Pool_ID',
                    function.pool_id,
                    global_step=self._num_samples
                )
        
        # Legal/Illegal function counts
        self._writer.add_scalars(
            'Functions/Count',
            {
                'legal': self._evaluate_success_program_num,
                'illegal': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        
        # Success rate
        total_funcs = self._evaluate_success_program_num + self._evaluate_failed_program_num
        if total_funcs > 0:
            success_rate = self._evaluate_success_program_num / total_funcs
            self._writer.add_scalar(
                'Functions/Success_Rate',
                success_rate,
                global_step=self._num_samples
            )
        
        # Timing information
        self._writer.add_scalars(
            'Time/Cumulative',
            {'sample': self._tot_sample_time, 'evaluate': self._tot_evaluate_time},
            global_step=self._num_samples
        )
        
        # Average time per sample
        if self._num_samples > 0:
            self._writer.add_scalars(
                'Time/Average_Per_Sample',
                {
                    'sample': self._tot_sample_time / self._num_samples,
                    'evaluate': self._tot_evaluate_time / self._num_samples
                },
                global_step=self._num_samples
            )
        
        # Log the function_str
        if self._cur_best_program_str:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )


    def _write_json(self, programs: code_manipulation.Function):
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score,
            'ood_score': programs.ood_score,
            'pool_id': programs.pool_id,
            'island_id': programs.island_id,
        }
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            self._record_and_verbose(sample_orders)
            self._write_tensorboard(programs)
            self._write_json(programs)
            
            # Track score history
            if programs.score is not None:
                self._score_history.append(programs.score)
            if programs.ood_score is not None:
                self._ood_score_history.append(programs.ood_score)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        ood_score = function.ood_score
        pool_id = function.pool_id
        
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'OOD Score    : {str(ood_score)}')
        print(f'Pool ID      : {str(pool_id)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str
            self._cur_best_program_ood_score = ood_score  # Track OOD score of best

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
    
    def update_pt_metrics(
        self,
        swap_rate_cumulative: float,
        swap_rate_rolling: float,
        xi: float,
        pool_stats: Optional[Dict[int, Dict[str, float]]] = None,
    ):
        """
        Update PT-specific metrics for tensorboard logging.
        
        Args:
            swap_rate_cumulative: Cumulative swap acceptance rate
            swap_rate_rolling: Rolling window swap acceptance rate
            xi: Current Boltzmann constant (ξ)
            pool_stats: Optional dict mapping pool_id to stats dict with keys:
                        'best_score', 'mean_score', 'diversity', 'size'
        """
        if not self._log_dir:
            return
        
        # Track swap rates
        self._pt_swap_rate_cumulative.append(swap_rate_cumulative)
        self._pt_swap_rate_rolling.append(swap_rate_rolling)
        self._pt_xi_values.append(xi)
        
        # Log to tensorboard
        if not np.isnan(swap_rate_cumulative):
            self._writer.add_scalar(
                'PT/Swap_Rate_Cumulative',
                swap_rate_cumulative,
                global_step=self._num_samples
            )
        if not np.isnan(swap_rate_rolling):
            self._writer.add_scalar(
                'PT/Swap_Rate_Rolling',
                swap_rate_rolling,
                global_step=self._num_samples
            )
        self._writer.add_scalar(
            'PT/Xi_Boltzmann',
            xi,
            global_step=self._num_samples
        )
        
        # Log per-pool statistics if provided
        if pool_stats:
            for pool_id, stats in pool_stats.items():
                if pool_id not in self._pt_pool_best_scores:
                    self._pt_pool_best_scores[pool_id] = []
                    self._pt_pool_diversity[pool_id] = []
                
                if 'best_score' in stats:
                    self._pt_pool_best_scores[pool_id].append(stats['best_score'])
                    self._writer.add_scalar(
                        f'PT_Pool_{pool_id}/Best_Score',
                        stats['best_score'],
                        global_step=self._num_samples
                    )
                
                if 'mean_score' in stats:
                    self._writer.add_scalar(
                        f'PT_Pool_{pool_id}/Mean_Score',
                        stats['mean_score'],
                        global_step=self._num_samples
                    )
                
                if 'diversity' in stats:
                    self._pt_pool_diversity[pool_id].append(stats['diversity'])
                    self._writer.add_scalar(
                        f'PT_Pool_{pool_id}/Diversity',
                        stats['diversity'],
                        global_step=self._num_samples
                    )
                
                if 'size' in stats:
                    self._writer.add_scalar(
                        f'PT_Pool_{pool_id}/Population_Size',
                        stats['size'],
                        global_step=self._num_samples
                    )
    
    def log_score_distribution(self):
        """Log score distribution statistics to tensorboard."""
        if not self._log_dir or not self._score_history:
            return
        
        scores = np.array(self._score_history[-100:])  # Last 100 samples
        
        self._writer.add_scalar(
            'Score_Distribution/Mean',
            np.mean(scores),
            global_step=self._num_samples
        )
        self._writer.add_scalar(
            'Score_Distribution/Std',
            np.std(scores),
            global_step=self._num_samples
        )
        self._writer.add_scalar(
            'Score_Distribution/Max',
            np.max(scores),
            global_step=self._num_samples
        )
        self._writer.add_scalar(
            'Score_Distribution/Min',
            np.min(scores),
            global_step=self._num_samples
        )
        
        # OOD score distribution
        if self._ood_score_history:
            ood_scores = np.array(self._ood_score_history[-100:])
            self._writer.add_scalar(
                'OOD_Score_Distribution/Mean',
                np.mean(ood_scores),
                global_step=self._num_samples
            )
            self._writer.add_scalar(
                'OOD_Score_Distribution/Std',
                np.std(ood_scores),
                global_step=self._num_samples
            )
    
    def get_best_program_info(self) -> Dict[str, Any]:
        """Return information about the current best program."""
        return {
            'score': self._cur_best_program_score,
            'ood_score': self._cur_best_program_ood_score,
            'sample_order': self._cur_best_program_sample_order,
            'program_str': self._cur_best_program_str,
        }
    
    def export_tracking_summary(self):
        """Export a summary of all tracked metrics to JSON."""
        if not self._log_dir:
            return
        
        summary = {
            'final_best_score': self._cur_best_program_score,
            'final_best_ood_score': self._cur_best_program_ood_score,
            'total_samples': self._num_samples,
            'success_rate': (
                self._evaluate_success_program_num / 
                (self._evaluate_success_program_num + self._evaluate_failed_program_num)
                if (self._evaluate_success_program_num + self._evaluate_failed_program_num) > 0
                else 0
            ),
            'total_sample_time': self._tot_sample_time,
            'total_evaluate_time': self._tot_evaluate_time,
            'score_history_stats': {
                'mean': float(np.mean(self._score_history)) if self._score_history else None,
                'std': float(np.std(self._score_history)) if self._score_history else None,
                'max': float(np.max(self._score_history)) if self._score_history else None,
                'min': float(np.min(self._score_history)) if self._score_history else None,
            },
            'ood_score_history_stats': {
                'mean': float(np.mean(self._ood_score_history)) if self._ood_score_history else None,
                'std': float(np.std(self._ood_score_history)) if self._ood_score_history else None,
                'max': float(np.max(self._ood_score_history)) if self._ood_score_history else None,
                'min': float(np.min(self._ood_score_history)) if self._ood_score_history else None,
            },
            'pt_metrics': {
                'final_xi': self._pt_xi_values[-1] if self._pt_xi_values else None,
                'swap_rate_cumulative': self._pt_swap_rate_cumulative[-1] if self._pt_swap_rate_cumulative else None,
                'swap_rate_rolling': self._pt_swap_rate_rolling[-1] if self._pt_swap_rate_rolling else None,
            }
        }
        
        path = os.path.join(self._log_dir, 'tracking_summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
