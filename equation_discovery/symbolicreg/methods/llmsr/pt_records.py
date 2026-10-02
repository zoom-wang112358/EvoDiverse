

"""Records for tracking PT swap operations in symbolic regression."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import csv
import json
import os


@dataclass
class SwapRecord:
    """One pairwise PT migration proposal and its outcome."""
    step: Optional[int]
    island_id: int  # Which island this swap occurred in
    pool_a_idx: int  # Cold pool index
    pool_b_idx: int  # Hot pool index
    idx_a: int  # Index of program in pool A
    idx_b: int  # Index of program in pool B
    beta1: float  # Inverse temperature of pool A (cold, higher beta)
    beta2: float  # Inverse temperature of pool B (hot, lower beta)
    score_a: float  # Score of program from pool A
    score_b: float  # Score of program from pool B
    prob_a: float  # Selection probability for program A
    prob_b: float  # Selection probability for program B
    logA: float  # Log acceptance probability
    logu: float  # Log uniform random sample
    accepted: bool  # Whether swap was accepted by PT criterion
    replaced_into_a: bool  # Whether program was actually placed into pool A
    replaced_into_b: bool  # Whether program was actually placed into pool B
    dup_block_a: bool  # Whether duplicate blocked placement into A
    dup_block_b: bool  # Whether duplicate blocked placement into B
    program_a_body: Optional[str]  # Body of program from A (before swap)
    program_b_body: Optional[str]  # Body of program from B (before swap)
    xi: Optional[float] = None  # Boltzmann constant (ξ) used for this swap


def export_swap_records_to_csv(records: list[SwapRecord], path: str, append: bool = False) -> None:
    """Export swap records to CSV file."""
    if not records:
        return
    
    mode = "a" if append else "w"
    file_exists = os.path.exists(path)
    
    with open(path, mode, newline="", encoding="utf-8") as f:
        fieldnames = list(asdict(records[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or not file_exists:
            writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))


def export_swap_records_to_jsonl(records: list[SwapRecord], path: str, append: bool = False) -> None:
    """Export swap records to JSONL file."""
    if not records:
        return
    
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


