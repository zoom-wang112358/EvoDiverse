from __future__ import annotations
import os, csv, json

def file_is_empty(path: str) -> bool:
    return (not os.path.exists(path)) or os.path.getsize(path) == 0

def write_row_csv(path: str, fieldnames: list, row_dict: dict):
    """
    Append a row with a fixed header. Missing fields are set to None.
    Extra fields are ignored.
    """
    header_needed = file_is_empty(path)
    aligned = {k: row_dict.get(k, None) for k in fieldnames}
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if header_needed:
            writer.writeheader()
        writer.writerow(aligned)

def write_row_jsonl(path: str, row_dict: dict):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row_dict, ensure_ascii=False) + '\n')
