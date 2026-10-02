import numpy as np

import selfies as sf
import yaml
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from rdkit import Chem
import numpy as np
import json
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
def export_swap_log_to_csv(self, path: str):
    if not hasattr(self, "swap_log") or not self.swap_log:
        print("No swap logs to export.")
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(self.swap_log[0]).keys()))
        writer.writeheader()
        for rec in self.swap_log:
            writer.writerow(asdict(rec))
    print(f"Swap log saved to {p}")

def export_swap_log_to_json(self, path: str):
    if not hasattr(self, "swap_log") or not self.swap_log:
        print("No swap logs to export.")
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in self.swap_log:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    print(f"Swap log saved to {p}")
def get_selfies_chars(selfies):
    """Obtain a list of all selfie characters in string selfies
    
    Parameters: 
    selfie (string) : A selfie string - representing a molecule 
    
    Example: 
    >>> get_selfies_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']
    
    Returns
    -------
    chars_selfies (list of strings) : 
        list of selfie characters present in molecule selfie
    """
    chars_selfies = sf.split_selfies(selfies)
    return list(chars_selfies)

def sanitize_smiles(smi):
    """Return canonical non-isomeric SMILES, or None for invalid input."""
    if smi == '':
        return None
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return smi_canon
    except:
        return None

def get_fp_scores(smiles_back, target_smi):
    """
    Given a list of SMILES (smiles_back), tanimoto similarities are calculated 
    (using Morgan fingerprints) to SMILES (target_smi). 
    Parameters
    ----------
    smiles_back : (list)
        List of valid SMILE strings. 
    target_smi : (str)
        Valid SMILES string. 
    Returns
    -------
    smiles_back_scores : (list of floats)
        List of fingerprint similarity scores of each smiles in input list. 
    """
    smiles_back_scores = []
    target = smi2mol(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = smi2mol(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

def from_yaml(work_dir, 
        fitness_function, 
        start_population,
        yaml_file, **kwargs):

    # create dictionary with parameters defined by yaml file 
    with open(yaml_file, 'r') as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params.update(kwargs)
    params.update({
        'work_dir': work_dir,
        'fitness_function': fitness_function,
        'start_population': start_population
    })

    return params



def mol2fp(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

def tanimoto_dist(fp1, fp2):
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)


def select_diverse_subset(mols, scores, max_size: int, score_weight: float = 0.0):
    n = len(mols)
    if n <= max_size:
        return list(range(n))


    fps = [mol2fp(m) for m in mols]

    start_idx = int(np.argmax(scores))
    selected = [start_idx]

    min_dists = np.array([
        tanimoto_dist(fps[i], fps[start_idx]) if i != start_idx else 0.0
        for i in range(n)
    ], dtype=float)

    while len(selected) < max_size:
        best_idx = None
        best_value = -1.0

        for i in range(n):
            if i in selected:
                continue
            div_score = min_dists[i]
            if score_weight > 0.0:
                div_score = div_score + score_weight * scores[i]

            if div_score > best_value:
                best_value = div_score
                best_idx = i

        if best_idx is None:
            break

        selected.append(best_idx)

        for j in range(n):
            if j in selected:
                continue
            d = tanimoto_dist(fps[j], fps[best_idx])
            if d < min_dists[j]:
                min_dists[j] = d

    return selected
