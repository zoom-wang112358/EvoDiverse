from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class SwapRecord:
    """One pairwise PT migration proposal and its outcome."""
    step: Optional[int]
    pool_a_idx: int
    pool_b_idx: int
    idx_a: int
    idx_b: int
    beta1: float
    beta2: float
    score_a: float
    score_b: float
    logA: float
    logu: float
    accepted: bool
    replaced_into_a: bool
    replaced_into_b: bool
    dup_block_a: bool
    dup_block_b: bool
    smi_a_old: Optional[str]
    smi_b_old: Optional[str]
    smi_a_new: Optional[str]
    smi_b_new: Optional[str]