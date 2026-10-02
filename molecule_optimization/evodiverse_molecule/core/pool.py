from __future__ import annotations
from multiprocessing import pool
from typing import List, Tuple, Optional, Union
from collections import defaultdict, deque
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from .math_utils import softmax_np, percentile_cap, scale_logw_for_01, power_score_probabilities
from .diversity import topk_smis_from_tuples, diversity, avg_knn_sim
from ..molecule_utils import select_diverse_subset

def _aggregate_parent_weight(pw: Optional[Union[float, List[float], Tuple[float, ...]]]) -> float:
    """Aggregate per-parent weights to one w_eff. Geometric mean is numerically stable."""
    if pw is None:
        return 1.0
    if isinstance(pw, (list, tuple, np.ndarray)):
        arr = np.asarray(pw, dtype=float)
        arr = np.clip(arr, 1e-12, 1e12)
        return float(np.exp(np.mean(np.log(arr))))
    try:
        val = float(pw)
        return float(np.clip(val, 1e-12, 1e12))
    except Exception:
        return 1.0

class Pool:
    def __init__(self, initial_mols: List[Mol], initial_scores: List[float],
                 inverse_temperature: float, population_size: int):
        self.inverse_temperature = float(inverse_temperature)
        self.population_size = int(population_size)

        init = sorted(list(zip(initial_scores, initial_mols)), key=lambda x: x[0], reverse=True)
        valid = [(s, m) for s, m in init if m is not None]
        self.mols = [t[1] for t in valid[: self.population_size]]
        self.scores = [t[0] for t in valid[: self.population_size]]

        self.swap_stats = defaultdict(lambda: {"proposals": 0, "accepts": 0})
        self.swap_hist = defaultdict(lambda: deque(maxlen=200))
        self.swap_stats_global = {"proposals": 0, "accepts": 0}
        self.swap_hist_global = deque(maxlen=200)

        self.parent_quota: Optional[int] = None
        self._parent_usage = np.zeros(len(self.mols), dtype=int)

    def __len__(self):
        return len(self.mols)

    # Score-weighted parent sampling with quotas and optional importance weights.
    def make_mating_pool(
        self,
        offspring_size: int,
        eps_mix: float = 0.0,             
        return_importance: bool = False,  
    ):
        all_tuples = list(zip(self.scores, self.mols))
        N = len(all_tuples)
        if offspring_size <= 0 or N == 0:
            return []
        
        scores = np.asarray(self.scores, dtype=float)
        base_w = scores
        sum_w = float(np.sum(base_w))
        p = (base_w / sum_w) if (np.isfinite(sum_w) and sum_w > 0.0) else np.ones(N, dtype=float) / N

        # per-parent quota bookkeeping
        if not hasattr(self, "_parent_usage") or len(self._parent_usage) != N:
            self._parent_usage = np.zeros(N, dtype=int)



        q = p.copy()
        Zg = float(np.sum(q))
        q = (q / Zg) if (np.isfinite(Zg) and Zg > 0.0) else np.ones(N, dtype=float) / N

        if eps_mix > 0.0:
            eps = float(eps_mix)
            q = (1.0 - eps) * q + eps * (1.0 / N)
            q = q / q.sum()

        # apply per-parent quota if configured
        if getattr(self, "parent_quota", None) is not None:
            quota = int(self.parent_quota)
            mask = (self._parent_usage < quota).astype(float)
            q = q * mask
            q = (q / q.sum()) if q.sum() > 0 else np.ones(N, dtype=float) / N

            idx_list = []
            usage = self._parent_usage.copy()
            q_work = q.copy()
            for _ in range(offspring_size):
                if q_work.sum() <= 0.0:
                    cand = np.where(usage < quota)[0]
                    if len(cand) == 0:
                        cand = np.arange(N)  # final fallback
                    pick = int(np.random.choice(cand))
                else:
                    pick = int(np.random.choice(N, p=q_work))

                idx_list.append(pick)
                usage[pick] += 1
                if usage[pick] >= quota:
                    q_work[pick] = 0.0
                    s = q_work.sum()
                    if s > 0:
                        q_work /= s
            mating_indices = np.array(idx_list, dtype=int)
            self._parent_usage = usage
        else:
            mating_indices = np.random.choice(N, p=q, size=offspring_size, replace=True)
            for i in mating_indices:
                self._parent_usage[i] += 1

        if not return_importance:
            return [all_tuples[i] for i in mating_indices]

        
        q_eff = np.clip(q, 1e-12, 1.0)
        p_eff = np.clip(p, 1e-12, 1.0)

        out = []
        for idx in mating_indices:
            s, m = all_tuples[idx]
            w_exact = float(p_eff[idx] / q_eff[idx])
            out.append((s, m, w_exact))
        return out

    
    def update(
        self,
        new_offspring_tuples: List[Tuple],
        lambda_reweight: float = 0.0,
        selection_mode: str = "rank",      
        parent_weight_key: Optional[str] = None,
        clip_c: float = 2.0,
        scale_k: float = 0.2,
        dynamic_clip: bool = False,
        dynamic_clip_range: Tuple[float,float] = (5.0, 95.0),
        keep_scores_in_01: bool = False,    # clip adjusted scores to [0,1] if desired
        alpha_div: float = 0.2,            # crowding penalty coefficient
    ):
        
        cand_raw_s, cand_mols, cand_w = [], [], []
        for s, m in zip(self.scores, self.mols):
            cand_raw_s.append(float(s))
            cand_mols.append(m)
            cand_w.append(1.0)

        for tup in new_offspring_tuples:
            if not tup or len(tup) < 2:
                continue
            s, m = tup[0], tup[1]
            w = 1.0
            if len(tup) >= 3:
                third = tup[2]
                if parent_weight_key is not None and isinstance(third, dict) and parent_weight_key in third:
                    w = _aggregate_parent_weight(third[parent_weight_key])
                else:
                    w = _aggregate_parent_weight(third)
            cand_raw_s.append(float(s))
            cand_mols.append(m)
            cand_w.append(float(w))

        
        seen = set()
        uniq_raw_s, uniq_mols, uniq_w, uniq_smi = [], [], [], []
        for s, m, w in zip(cand_raw_s, cand_mols, cand_w):
            if m is None:
                continue
            smi = Chem.MolToSmiles(m)
            if smi in seen:
                continue
            seen.add(smi)
            uniq_raw_s.append(s)
            uniq_mols.append(m)
            uniq_w.append(np.clip(w, 1e-12, 1e12))
            uniq_smi.append(smi)

        if not uniq_raw_s:
            return

        uniq_raw_s = np.asarray(uniq_raw_s, dtype=float)
        uniq_logw  = np.log(np.asarray(uniq_w, dtype=float))

        
        use_clip_c = float(clip_c)
        if dynamic_clip and len(uniq_logw) >= 10:
            use_clip_c = percentile_cap(uniq_logw, *dynamic_clip_range)
        scaled_logw = scale_logw_for_01(uniq_logw, clip_c=use_clip_c, scale_k=scale_k)

        
        adj_scores = uniq_raw_s + float(lambda_reweight) * scaled_logw
        if keep_scores_in_01:
            adj_scores = np.clip(adj_scores, 0.0, 1.0)

        
        if self.inverse_temperature < 0.6:
            if len(uniq_mols) >= 2 and alpha_div > 0.0:
                knn_sim = avg_knn_sim(uniq_mols, k=10)
                adj_scores = adj_scores - float(alpha_div) * knn_sim
        else:
            adj_scores = adj_scores  
        # select next population
        n = min(self.population_size, len(adj_scores))
        if self.inverse_temperature > 0.5:
            
            elite_k = min(3, n // 10)
            order = np.argsort(-adj_scores)
            elite_idx = list(order[:elite_k])  

            
            rest_idx = order[elite_k:]
            need = n - elite_k
            take = elite_idx.copy()
            if selection_mode == "rank":
                take = np.argsort(-adj_scores)[:n]
            else:
                
                rest_scores = adj_scores[rest_idx]
                probs = power_score_probabilities(rest_scores, inverse_temperature=self.inverse_temperature)
                pick = np.random.choice(len(rest_idx), size=need, replace=False, p=probs)
                take += [int(rest_idx[i]) for i in pick]
        elif self.inverse_temperature > 0.1:
            
            elite_k = min(3, n // 10)
            order = np.argsort(-adj_scores)
            elite_idx = list(order[:elite_k])  

            
            rest_idx = order[elite_k:]
            need = n - elite_k
            take = elite_idx.copy()
            if selection_mode == "rank":
                take = np.argsort(-adj_scores)[:n]
            else:
                # Score-power sampling over non-elite candidates
                rest_scores = adj_scores[rest_idx]
                probs = power_score_probabilities(rest_scores, inverse_temperature=self.inverse_temperature)
                pick = np.random.choice(len(rest_idx), size=need, replace=False, p=probs)
                take += [int(rest_idx[i]) for i in pick]

        else:
            
            order = np.argsort(-adj_scores)

            
            rest_idx = order
            need = n
            take = []
            if selection_mode == "rank":
                take = np.argsort(-adj_scores)[:n]
            else:
                # Score-power sampling over non-elite candidates
                rest_scores = adj_scores[rest_idx]
                probs = power_score_probabilities(rest_scores, inverse_temperature=self.inverse_temperature)
                pick = np.random.choice(len(rest_idx), size=need, replace=False, p=probs)
                take += [int(rest_idx[i]) for i in pick]


        self.mols   = [uniq_mols[i] for i in take]
        self.scores = [float(uniq_raw_s[i]) for i in take]  # keep raw scores for logging

        self._parent_usage = np.zeros(len(self.mols), dtype=int)

    # ---------- Utilities ----------
    def get_best_individuals(self, num_to_get: int) -> List[Tuple[float, Mol]]:
        if num_to_get <= 0: 
            return []
        sorted_tuples = sorted(zip(self.scores, self.mols), key=lambda x: x[0], reverse=True)
        return sorted_tuples[:num_to_get]

    def replace_worst_individuals(self, incoming_tuples: List[Tuple[float, Mol]]):
        """Replace the lowest-scoring members with incoming candidates."""
        num_to_replace = len(incoming_tuples)
        if num_to_replace == 0:
            return
        current = list(zip(self.scores, self.mols))
        sorted_current = sorted(current, key=lambda x: x[0])  # ascending
        kept = sorted_current[num_to_replace:]
        new_population = kept + incoming_tuples
        # reuse update to dedup & select
        self.update(new_population)

    def top_k_smis(self, k: int = 100) -> List[str]:
        tuples = list(zip(self.scores, self.mols))
        return topk_smis_from_tuples(tuples, k)

    def top_k_diversity(self, k: int = 100) -> float:
        return diversity(list(zip(self.scores, self.mols)), k=k)

    def prune_for_diversity(self, keep_size=120, elites=20, score_weight: float = 0.0):
        """
        Reduce size to keep_size by preserving elites and filling the rest via
        diversity-aware subset selection.
        """
        n = len(self.mols)
        if n <= keep_size:
            return
        idx = np.argsort(self.scores)[::-1]
        elite_idx = idx[:elites]
        rest_idx  = idx[elites:]
        chosen_rel = select_diverse_subset(
            [self.mols[i] for i in rest_idx],
            [self.scores[i] for i in rest_idx],
            max_size=keep_size - elites,
            score_weight=score_weight
        )
        chosen_idx = list(elite_idx) + [rest_idx[i] for i in chosen_rel]
        self.mols   = [self.mols[i] for i in chosen_idx]
        self.scores = [self.scores[i] for i in chosen_idx]
