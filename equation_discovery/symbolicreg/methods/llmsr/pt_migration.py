

"""
Migration logic for Parallel Tempering between pools.

Implements the PT swap criterion where candidates are exchanged between
cold (exploitation) and hot (exploration) pools based on Metropolis-Hastings
acceptance probability.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from llmsr.pt_pool import PTPool
from llmsr.pt_records import SwapRecord
from llmsr.pt_math_utils import softmax_probs, compute_logA_from_rewards


def pt_migration_softmax_random_inplace(
    pool_a: PTPool,
    pool_b: PTPool,
    migration_size: int,
    island_id: int = 0,
    pool_a_idx: int = 0,
    pool_b_idx: int = 1,
    step: Optional[int] = None,
    xi: float = 2.5,
) -> Tuple[int, int, int, List[SwapRecord]]:
    """
    Perform PT migration between two pools using softmax sampling.
    
    Sample indices from each pool with Softmax(β·score) without replacement;
    apply PT swap criterion pairwise; if accepted, replace in-place.
    
    Args:
        pool_a: First pool (typically cold, high beta)
        pool_b: Second pool (typically hot, low beta)
        migration_size: Number of swap attempts
        island_id: ID of the island containing these pools
        pool_a_idx: Index of pool A (for logging)
        pool_b_idx: Index of pool B (for logging)
        step: Current global step (for logging)
        xi: Boltzmann constant (ξ) for computing acceptance probability.
            This is adaptively adjusted to maintain target swap rate.
        
    Returns:
        Tuple of (proposals, accepted_into_a, accepted_into_b, records)
    """
    k = int(min(migration_size, len(pool_a), len(pool_b)))
    if k <= 0:
        return 0, 0, 0, []
    
    beta1 = float(pool_a.temperature)  # Cold pool (higher beta)
    beta2 = float(pool_b.temperature)  # Hot pool (lower beta)
    
    # Compute selection probabilities
    probs_a = softmax_probs(pool_a.scores, beta1)
    probs_b = softmax_probs(pool_b.scores, beta2)
    
    # Sample indices without replacement
    idx_a = np.random.choice(len(pool_a), size=k, replace=False, p=probs_a)
    idx_b = np.random.choice(len(pool_b), size=k, replace=False, p=probs_b)
    
    # Get program bodies for deduplication
    bodies_a = pool_a.get_program_bodies()
    bodies_b = pool_b.get_program_bodies()
    
    proposals = 0
    acc_into_a = 0
    acc_into_b = 0
    records: List[SwapRecord] = []
    
    for ia, ib in zip(idx_a, idx_b):
        ia, ib = int(ia), int(ib)
        
        # Get programs and scores
        prog_a = pool_a.programs[ia]
        score_a = pool_a.scores[ia]
        scores_per_test_a = pool_a.scores_per_test[ia]
        
        prog_b = pool_b.programs[ib]
        score_b = pool_b.scores[ib]
        scores_per_test_b = pool_b.scores_per_test[ib]
        
        # Get OOD scores if available
        ood_score_a = pool_a.ood_scores[ia] if ia < len(pool_a.ood_scores) else None
        ood_scores_per_test_a = pool_a.ood_scores_per_test[ia] if ia < len(pool_a.ood_scores_per_test) else None
        ood_score_b = pool_b.ood_scores[ib] if ib < len(pool_b.ood_scores) else None
        ood_scores_per_test_b = pool_b.ood_scores_per_test[ib] if ib < len(pool_b.ood_scores_per_test) else None
        
        if prog_a is None or prog_b is None:
            continue
        
        # Compute PT acceptance probability using adaptive xi
        logA = compute_logA_from_rewards(score_a, score_b, beta1, beta2, xi=xi)
        logu = float(np.log(np.random.rand()))
        accepted = (logu <= min(0.0, logA))
        proposals += 1
        
        body_a_old = prog_a.body.strip()
        body_b_old = prog_b.body.strip()
        body_a_new = prog_b.body.strip()  # What would go into A
        body_b_new = prog_a.body.strip()  # What would go into B
        
        replaced_a = False
        replaced_b = False
        dup_a = False
        dup_b = False
        
        if accepted:
            # Check for duplicates before swapping
            # De-dup: check if new body already exists in target pool (excluding current slot)
            
            # For pool A: check if prog_b's body already exists (excluding slot ia)
            set_a = set(bodies_a)
            if body_a_old in set_a:
                set_a.discard(body_a_old)
            
            if body_a_new not in set_a:
                # Swap prog_b into pool_a at index ia (including OOD scores)
                pool_a.replace_at_index(
                    ia, prog_b, score_b, scores_per_test_b,
                    ood_score=ood_score_b, ood_scores_per_test=ood_scores_per_test_b
                )
                bodies_a[ia] = body_a_new
                acc_into_a += 1
                replaced_a = True
            else:
                dup_a = True
            
            # For pool B: check if prog_a's body already exists (excluding slot ib)
            set_b = set(bodies_b)
            if body_b_old in set_b:
                set_b.discard(body_b_old)
            
            if body_b_new not in set_b:
                # Swap prog_a into pool_b at index ib (including OOD scores)
                pool_b.replace_at_index(
                    ib, prog_a, score_a, scores_per_test_a,
                    ood_score=ood_score_a, ood_scores_per_test=ood_scores_per_test_a
                )
                bodies_b[ib] = body_b_new
                acc_into_b += 1
                replaced_b = True
            else:
                dup_b = True
        
        # Create record
        rec = SwapRecord(
            step=None if step is None else int(step),
            island_id=int(island_id),
            pool_a_idx=int(pool_a_idx),
            pool_b_idx=int(pool_b_idx),
            idx_a=ia,
            idx_b=ib,
            beta1=float(beta1),
            beta2=float(beta2),
            score_a=float(score_a),
            score_b=float(score_b),
            prob_a=float(probs_a[ia]),
            prob_b=float(probs_b[ib]),
            logA=float(logA),
            logu=float(logu),
            accepted=bool(accepted),
            replaced_into_a=bool(replaced_a),
            replaced_into_b=bool(replaced_b),
            dup_block_a=bool(dup_a),
            dup_block_b=bool(dup_b),
            program_a_body=body_a_old[:200] if body_a_old else None,  # Truncate for logging
            program_b_body=body_b_old[:200] if body_b_old else None,
            xi=float(xi),
        )
        records.append(rec)
    
    return proposals, acc_into_a, acc_into_b, records


def compute_swap_rates(
    swap_stats: dict,
    swap_hist: dict,
    key: Tuple[int, int],
) -> Tuple[float, float]:
    """
    Compute cumulative and rolling swap rates for a pool pair.
    
    Args:
        swap_stats: Dictionary tracking proposals and accepts per pair
        swap_hist: Deque of recent accept/reject outcomes per pair
        key: Tuple of (pool_a_idx, pool_b_idx)
        
    Returns:
        Tuple of (cumulative_rate, rolling_rate)
    """
    pair_total = swap_stats[key]["proposals"]
    pair_acc = swap_stats[key]["accepts"]
    
    cum_rate = (pair_acc / pair_total) if pair_total > 0 else float("nan")
    roll_rate = (sum(swap_hist[key]) / len(swap_hist[key])) if len(swap_hist[key]) > 0 else float("nan")
    
    return cum_rate, roll_rate


