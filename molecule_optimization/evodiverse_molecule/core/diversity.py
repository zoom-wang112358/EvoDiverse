from __future__ import annotations
from typing import List, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import tdc

def topk_avg(scores: List[float], k: int = 10) -> float:
    if not scores:
        return float('nan')
    k = min(len(scores), k)
    return float(np.mean(sorted(scores, reverse=True)[:k]))

def topk_smis_from_tuples(score_mol_tuples: List[Tuple[float, 'Chem.Mol']], k: int) -> List[str]:
    """Pick top-k unique SMILES by score (descending)."""
    sorted_tuples = sorted(score_mol_tuples, key=lambda x: x[0], reverse=True)
    smis, seen = [], set()
    for s, m in sorted_tuples:
        if m is None:
            continue
        smi = Chem.MolToSmiles(m)
        if not smi or smi in seen:
            continue
        seen.add(smi)
        smis.append(smi)
        if len(smis) >= k:
            break
    return smis

def diversity_from_smis(smis: List[str]) -> float:
    if not smis:
        return 0.0
    diversity_evaluator = tdc.Evaluator(name='Diversity')
    val = diversity_evaluator(smis)
    if isinstance(val, dict):
        return float(val.get('diversity', next(iter(val.values()))))
    return float(val)

def diversity(score_mol_tuples: List[Tuple[float, 'Chem.Mol']], k: int = 100) -> float:
    """Compute Diversity(TDC) on top-k unique SMILES by score."""
    smis_topk = topk_smis_from_tuples(score_mol_tuples, k)
    return diversity_from_smis(smis_topk)

def ecfp4(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)

def avg_knn_sim(mols, k=10):
    """Average top-k Tanimoto similarity for each molecule (crowding proxy)."""
    fps = [ecfp4(m) for m in mols]
    n = len(fps)
    sims = np.zeros(n, dtype=float)
    for i in range(n):
        arr = []
        for j in range(n):
            if i == j:
                continue
            v = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            arr.append(v)
        arr.sort(reverse=True)
        take = arr[:min(k, len(arr))]
        sims[i] = np.mean(take) if take else 0.0
    return sims
