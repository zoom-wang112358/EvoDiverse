from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from rdkit import Chem
from .math_utils import softmax_probs, compute_logA_from_rewards
from .records import SwapRecord

def _smi(m): 
    return Chem.MolToSmiles(m) if m is not None else None

def swap_between_pools(pool_a, pool_b, migration_size: int,
                                        pool_a_idx: int, pool_b_idx: int,
                                        step: Optional[int] = None, coefficient: float = 1.0):
    """
    Sample swap candidates without replacement and apply the PT criterion.
    Hot pools use softmax(score); cold pools use uniform sampling with elite
    protection. Accepted candidates replace molecules in-place, excluding duplicates.
    Returns: (proposals, accepted_into_a, accepted_into_b, records)
    """

    def build_candidates(pool, elite_guard: bool, sample_mode: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
        n = len(pool)
        if n == 0:
            return np.array([], dtype=int), np.array([])


        cand = np.arange(n, dtype=int)


        if elite_guard:
            start = min(3, n)
            cand = np.arange(start, n, dtype=int)


        if cand.size == 0:
            return np.array([], dtype=int), np.array([])


        if sample_mode == "uniform":
            probs_full = np.ones(n, dtype=float) / n
        else:  # "softmax"
            probs_full = softmax_probs(pool.scores, 1)

        probs_cand = probs_full[cand]
        s = probs_cand.sum()
        if not np.isfinite(s) or s <= 0.0:
            probs_cand = np.ones_like(probs_cand, dtype=float) / len(probs_cand)
        else:
            probs_cand = probs_cand / s
        return cand, probs_cand


    is_cold_a = (float(pool_a.inverse_temperature) > 0.5)
    is_cold_b = (float(pool_b.inverse_temperature) > 0.5)


    is_hot_a = (float(pool_a.inverse_temperature) < 0.6)
    is_hot_b = (float(pool_b.inverse_temperature) < 0.6)

    cand_a, probs_a = build_candidates(pool_a, elite_guard=is_cold_a, sample_mode="softmax" if is_hot_a else "uniform")
    cand_b, probs_b = build_candidates(pool_b, elite_guard=is_cold_b, sample_mode="softmax" if is_hot_b else "uniform")


    if cand_a.size == 0 or cand_b.size == 0:
        return 0, 0, 0, []

    k = int(min(migration_size, cand_a.size, cand_b.size))
    if k <= 0:
        return 0, 0, 0, []


    idx_a = np.random.choice(cand_a, size=k, replace=False, p=probs_a)
    idx_b = np.random.choice(cand_b, size=k, replace=False, p=probs_b)

    beta1 = float(pool_a.inverse_temperature)
    beta2 = float(pool_b.inverse_temperature)


    smiles_a = [_smi(m) for m in pool_a.mols]
    smiles_b = [_smi(m) for m in pool_b.mols]

    proposals = 0
    acc_into_a = 0
    acc_into_b = 0
    records: List[SwapRecord] = []

    for ia, ib in zip(idx_a, idx_b):
        ma, sa = pool_a.mols[ia], pool_a.scores[ia]
        mb, sb = pool_b.mols[ib], pool_b.scores[ib]
        if (ma is None) or (mb is None):
            continue

        logA = compute_logA_from_rewards(sa, sb, beta1, beta2, coefficient)
        logu = float(np.log(np.random.rand()))
        accepted = (logu <= min(0.0, logA))
        proposals += 1

        smi_a_old = smiles_a[ia]
        smi_b_old = smiles_b[ib]
        smi_a_new = _smi(mb)
        smi_b_new = _smi(ma)
        replaced_a = False
        replaced_b = False
        dup_a = False
        dup_b = False

        if accepted:
            # de-dup against the pool except the current slot
            if smi_a_new is not None:
                set_a = set(smiles_a)
                if smi_a_old in set_a:
                    set_a.remove(smi_a_old)
                if smi_a_new not in set_a:
                    pool_a.mols[ia] = mb
                    pool_a.scores[ia] = sb
                    smiles_a[ia] = smi_a_new
                    acc_into_a += 1
                    replaced_a = True
                else:
                    dup_a = True

            if smi_b_new is not None:
                set_b = set(smiles_b)
                if smi_b_old in set_b:
                    set_b.remove(smi_b_old)
                if smi_b_new not in set_b:
                    pool_b.mols[ib] = ma
                    pool_b.scores[ib] = sa
                    smiles_b[ib] = smi_b_new
                    acc_into_b += 1
                    replaced_b = True
                else:
                    dup_b = True

        rec = SwapRecord(
            step=None if step is None else int(step),
            pool_a_idx=int(pool_a_idx),
            pool_b_idx=int(pool_b_idx),
            idx_a=int(ia),
            idx_b=int(ib),
            beta1=float(beta1),
            beta2=float(beta2),
            score_a=float(sa),
            score_b=float(sb),
            logA=float(logA),
            logu=float(logu),
            accepted=bool(accepted),
            replaced_into_a=bool(replaced_a),
            replaced_into_b=bool(replaced_b),
            dup_block_a=bool(dup_a),
            dup_block_b=bool(dup_b),
            smi_a_old=smi_a_old,
            smi_b_old=smi_b_old,
            smi_a_new=smi_a_new,
            smi_b_new=smi_b_new,
        )
        records.append(rec)

    return proposals, acc_into_a, acc_into_b, records
