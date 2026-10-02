from __future__ import annotations
import os, time, json
from typing import List, Optional
from collections import defaultdict, deque
from dataclasses import asdict
import numpy as np
from rdkit import Chem

from .core.pool import Pool
from .core.records import SwapRecord
from .core.migration import swap_between_pools
from .core.diversity import topk_avg
from .logging_utils import write_row_csv, write_row_jsonl
from rdkit.Chem import AllChem, DataStructs
from . import crossover as co, mutation as mu
from .base_optimizer import BaseOptimizer, top_auc
from .llm_proposer import LLMMoleculeProposer

class EvoDiverseOptimizer(BaseOptimizer):
    """Optimize molecular scores with cold and hot populations and PT swaps."""
    def __init__(self, args=None):
        super().__init__(args)
        self.method_name = "evodiverse"
        self.args = args
        self.coefficient = 2.5
        self.proposer = LLMMoleculeProposer()
        if self.proposer is not None:
            self.proposer.task = self.args.oracles

        self.inverse_temperatures = getattr(args, 'inverse_temperatures', [0.8, 0.2])
        self.num_pools = len(self.inverse_temperatures)
        self.pools: List[Pool] = []
        print(f"Initializing EvoDiverse with {self.num_pools} pools. Inverse temperatures: {self.inverse_temperatures}")
        
        # Migration parameters
        self.migration_probability = getattr(args, 'migration_probability', 0.1)
        self.migration_size = getattr(args, 'migration_size', 30)

        self.swap_stats = defaultdict(lambda: {"proposals": 0, "accepts": 0})
        self.swap_hist = defaultdict(lambda: deque(maxlen=200))
        self.swap_stats_global = {"proposals": 0, "accepts": 0}
        self.swap_hist_global = deque(maxlen=200)
        self.swap_log: List[SwapRecord] = []
        self.global_step = 0

        # Logging config
        self.log_every_migrations = getattr(args, 'log_every_migrations', 1)
        self.log_dir = getattr(args, 'rate_dir', './rates')
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, args.oracles[0] + 'pt_metrics.csv')
        self.jsonl_path = os.path.join(self.log_dir, args.oracles[0] + 'pt_metrics.jsonl')

        self.seed = getattr(args, "seed", None) 

        self.poplog_dir = getattr(args, "poplog_dir", self.log_dir)
        os.makedirs(self.poplog_dir, exist_ok=True)

        self.pop_csv_path   = os.path.join(self.poplog_dir, args.oracles[0] + "_population_top10.csv")
        self.pop_jsonl_path = os.path.join(self.poplog_dir, args.oracles[0] + "_population_top10.jsonl")

        self.pop_csv_fieldnames = [
            "timestamp",
            "oracle",
            "seed",
            "generation",
            "global_step",
            "oracle_calls",         
            "pool_idx",
            "pool_inverse_temperature",
            "pool_size",
            "top10_avg",
            "best",
        ]

        base_fields = [
            "timestamp", "migration_step",
            "pair", "pair_rate_cum", "pair_rate_roll",
            "global_rate_cum", "global_rate_roll",
        ]



        pool_fields = []
        for i in range(self.num_pools):
            pool_fields += [f"pool{i}_beta", f"pool{i}_top10_avg", f"pool{i}_top100_diversity", f"pool{i}_best", f"pool{i}_size"]
        self.csv_fieldnames = base_fields + pool_fields

        self.metrics_log = []
        self.migration_step = 0

    def _snapshot_and_save_metrics(self, pair_key, pair_rate_cum, pair_rate_roll, glob_rate_cum, glob_rate_roll):
        ts = time.time()
        per_pool = {}
        for i, pool in enumerate(self.pools):
            top10 = topk_avg(pool.scores, k=10)
            best = max(pool.scores) if pool.scores else float('nan')
            per_pool[f'pool{i}_beta'] = float(pool.inverse_temperature)
            per_pool[f'pool{i}_top10_avg'] = float(top10)
            per_pool[f'pool{i}_top100_diversity'] = float(pool.top_k_diversity(k=100))
            per_pool[f'pool{i}_best'] = float(best)
            per_pool[f'pool{i}_size'] = int(len(pool.scores))

        record = {
            "timestamp": ts,
            "migration_step": self.migration_step,
            "pair": f"{pair_key[0]}-{pair_key[1]}",
            "pair_rate_cum": float(pair_rate_cum),
            "pair_rate_roll": float(pair_rate_roll),
            "global_rate_cum": float(glob_rate_cum),
            "global_rate_roll": float(glob_rate_roll),
            "coefficient": float(self.coefficient)
        }
        record.update(per_pool)

        self.metrics_log.append(record)
        write_row_csv(self.csv_path, self.csv_fieldnames, record)
        write_row_jsonl(self.jsonl_path, record)

    def _perform_migration(self):
        if self.num_pools < 2:
            return
        pool_a_idx, pool_b_idx = np.random.choice(self.num_pools, size=2, replace=False)
        pool_a = self.pools[pool_a_idx]
        pool_b = self.pools[pool_b_idx]
        step = getattr(self, "global_step", None)
        if len(self.metrics_log) > 0:
            if self.metrics_log[-1]["pair_rate_cum"] < 0.3:
                self.coefficient = self.coefficient * 0.9
            elif self.metrics_log[-1]["pair_rate_cum"] > 0.5:
                self.coefficient = self.coefficient * 1.1
        proposed, acc_a, acc_b, records = swap_between_pools(
            pool_a, pool_b, self.migration_size,
            pool_a_idx=pool_a_idx, pool_b_idx=pool_b_idx, step=step, coefficient=self.coefficient
        )
        if not hasattr(self, "swap_log"):
            self.swap_log: List[SwapRecord] = []
        self.swap_log.extend(records)

        accepted_total = acc_a  # by symmetry
        key = (pool_a_idx, pool_b_idx)
        self.swap_stats[key]["proposals"] += proposed
        self.swap_stats[key]["accepts"]   += accepted_total
        self.swap_stats_global["proposals"] += proposed
        self.swap_stats_global["accepts"]   += accepted_total

        self.swap_hist[key].extend([1] * accepted_total)
        self.swap_hist[key].extend([0] * (proposed - accepted_total))
        self.swap_hist_global.extend([1] * accepted_total)
        self.swap_hist_global.extend([0] * (proposed - accepted_total))

        pair_total = self.swap_stats[key]["proposals"]
        pair_acc   = self.swap_stats[key]["accepts"]
        pair_rate_cum  = (pair_acc / pair_total) if pair_total > 0 else float("nan")
        pair_rate_roll = (sum(self.swap_hist[key]) / len(self.swap_hist[key])) if len(self.swap_hist[key]) > 0 else float("nan")

        glob_total = self.swap_stats_global["proposals"]
        glob_acc   = self.swap_stats_global["accepts"]
        glob_rate_cum  = (glob_acc / glob_total) if glob_total > 0 else float("nan")
        glob_rate_roll = (sum(self.swap_hist_global) / len(self.swap_hist_global)) if len(self.swap_hist_global) > 0 else float("nan")

        print(
            f"--- PT migration between Pool {pool_a_idx} (β={pool_a.inverse_temperature:.3g}) "
            f"and Pool {pool_b_idx} (β={pool_b.inverse_temperature:.3g}); "
            f"this_step: proposed={proposed}, accepted={accepted_total}, "
            f"pair_cum={pair_rate_cum:.3f}, global_cum={glob_rate_cum:.3f} ---"
        )

        self.migration_step += 1
        if (self.migration_step % self.log_every_migrations) == 0:
            self._snapshot_and_save_metrics(
                pair_key=key,
                pair_rate_cum=pair_rate_cum,
                pair_rate_roll=pair_rate_roll,
                glob_rate_cum=glob_rate_cum,
                glob_rate_roll=glob_rate_roll
            )

    def _optimize(self, oracle, config):

        self.popdump_root = getattr(self.args, "popdump_root", "./population_dumps")
        self.popdump_every = int(getattr(self.args, "popdump_every", 1))   
        self.popdump_mode = getattr(self.args, "popdump_mode", "per_round")  
        self.popdump_gzip = bool(getattr(self.args, "popdump_gzip", False))  

        oracle_name = self.args.oracles[0] if getattr(self.args, "oracles", None) else "oracle"
        seed_tag = f"seed_{self.seed}" if self.seed is not None else "seed_unknown"
        self.popdump_dir = os.path.join(self.popdump_root, oracle_name, seed_tag)
        os.makedirs(self.popdump_dir, exist_ok=True)

        self.popdump_append_path = os.path.join(self.popdump_dir, "population_all_rounds.jsonl")


        self.migration_step = 0
        self.oracle.assign_evaluator(oracle)
        
        for i in range(self.num_pools):
            print(f"Initializing Pool {i} with inverse temperature {self.inverse_temperatures[i]:.3g}")
            # 1) create initial population
            if self.smi_file is not None:
                starting_population = self.all_smiles[:config["initial_population_size"]]
            else:
                starting_population = np.random.choice(self.all_smiles, config["initial_population_size"])
            initial_smiles = list(starting_population)
            initial_mols = [Chem.MolFromSmiles(s) for s in initial_smiles]
            initial_scores = self.oracle([Chem.MolToSmiles(mol) for mol in initial_mols])
            self.pools.append(Pool(initial_mols, initial_scores, self.inverse_temperatures[i], config["population_size"]))

        for pool in self.pools:
            pool.parent_quota = 30 if pool.inverse_temperature < 0.6 else 40

        patience = 0
        round_ = 0
        while not self.finish:
            self.global_step += 1
            print(f"=== Generation {round_} ===")

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
            else:
                old_score = 0

            all_offspring = []
            for i, pool in enumerate(self.pools):
                mating_tuples = pool.make_mating_pool(
                    config["offspring_size"], eps_mix=0.02, return_importance=True
                )
                if not mating_tuples:
                    continue

                if pool.inverse_temperature < 0.3:
                    temp = 1.3
                    mode = 'hot'
                else:
                    temp = 1.3
                    mode = 'cold'

                def _propose_once():
                    res = self.proposer.propose(mating_tuples, config["mutation_rate"], temp, mode)
                    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
                        mol, meta = res
                        pws = meta.get("parent_weights", None)
                    else:
                        mol, meta, pws = res, None, None
                    return mol, pws

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                    futs = [executor.submit(_propose_once) for _ in range(config["offspring_size"])]
                    for fu in futs:
                        mol, parent_weights = fu.result()
                        if mol is None:
                            continue
                        all_offspring.append((mol, i, parent_weights))

            if not all_offspring:
                continue

            mols_only = [m for (m, _, _) in all_offspring]
            sanitized_mols = self.sanitize(mols_only)
            if not sanitized_mols:
                continue

            valid_triplets = [(m, origin, pws)
                            for (m, (orig_m, origin, pws)) in zip(sanitized_mols, all_offspring)
                            if m is not None]
            if not valid_triplets:
                continue

            offspring_scores = []
            for m, origin, _ in valid_triplets:
                smi = Chem.MolToSmiles(m)
                temp_tag = self.pools[origin].inverse_temperature
                score = self.oracle(smi, temp_tag)
                offspring_scores.append(score)


            # 3) dispatch back to source pools with optional parent weights
            new_tuples_by_pool = {i: [] for i in range(self.num_pools)}
            for (score, (mol, origin_idx, parent_weights)) in zip(offspring_scores, valid_triplets):
                meta = {"parent_weights": parent_weights} if parent_weights is not None else {}
                if meta:
                    new_tuples_by_pool[origin_idx].append((score, mol, meta))
                else:
                    new_tuples_by_pool[origin_idx].append((score, mol))

            # 4) pool-wise update with IS correction + diversity penalty
            lambda_rw = float(config.get("importance_reweight", 0.0))
            alpha_div = float(config.get("alpha_diversity", 0.15))

            for i, pool in enumerate(self.pools):
                if pool.inverse_temperature <= 0.6:
                    sel_mode = "sample" 
                else:
                    sel_mode = config.get("selection_mode", "sample")
                pool.update(
                    new_tuples_by_pool[i],
                    lambda_reweight=lambda_rw,
                    selection_mode=sel_mode,
                    parent_weight_key="parent_weights",
                    alpha_div=alpha_div,
                )

            # 5) schedule migration by phase
            if   round_ <= 15:  dividend, self.migration_size = 3, 40
            elif round_ <= 35:  dividend, self.migration_size = 4, 30
            else:               dividend, self.migration_size = 4, 30

            if (round_ + 1) % dividend == 0:
                self._perform_migration()
                self.export_swap_log_to_csv("logs/swap_log.csv", append=True)
                self.export_swap_log_to_json("logs/swap_log.jsonl", append=True)
                self.swap_log = []
            # 4.5) log per-pool population stats after update (top10 avg etc.)
            self._log_population_top10_after_update(generation=round_)
            self._dump_population_this_round(generation=round_)
            round_ += 1
            print(f"=== End of generation {round_}, total evaluated: {len(self.oracle)} ===")

            # 6) early stopping
            if len(self.oracle) > 100:
                self.sort_buffer()
                new_score = np.mean([item[1][0] for item in list(self.mol_buffer.items())[:100]])
                if (new_score - old_score) < 1e-3:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('Convergence criteria met. Stopping.')
                        break
                else:
                    patience = 0
                old_score = new_score

            if self.finish:
                self.log_intermediate(finish=True)
                break
    def report_all_pools_metrics(
        self,
        topk: int = 10,
        div_k: int = 100,
        auc_freq_log: int = 50,
        auc_max_calls: Optional[int] = None,
        finish: bool = True,
        temp_tol: float = 1e-8,
    ):
        out = {}
        buf = getattr(self, "mol_buffer", None)
        if buf is None or len(buf) == 0:
            buf = getattr(self.oracle, "mol_buffer", {})
        if auc_max_calls is None:
            auc_max_calls = max(1, len(buf))

        for i, pool in enumerate(self.pools):
            scores = list(pool.scores)
            topk_avg = float(np.mean(sorted(scores, reverse=True)[:min(topk, len(scores))])) if scores else 0.0
            best     = float(np.max(scores)) if scores else 0.0
            div_val  = float(pool.top_k_diversity(k=div_k)) 

            auc_i = 0.0
            if buf:
                auc_i = float(top_auc(
                    buffer=buf,
                    top_n=topk,
                    finish=finish,
                    freq_log=auc_freq_log,
                    max_oracle_calls=auc_max_calls,
                    temp=float(pool.inverse_temperature),
                    temp_tol=temp_tol
                ))

            out.update({
                f"pool{i}_beta": float(pool.inverse_temperature),
                f"pool{i}_size": int(len(pool.mols)),
                f"pool{i}_top{topk}_avg_finalpop": topk_avg,
                f"pool{i}_best_finalpop": best,
                f"pool{i}_diversity_top{div_k}_finalpop": div_val,
                f"pool{i}_top{topk}_AUC": auc_i,
            })

        if buf:
            out[f"overall_top{topk}_AUC"] = float(top_auc(
                buffer=buf,
                top_n=topk,
                finish=finish,
                freq_log=auc_freq_log,
                max_oracle_calls=auc_max_calls,
                temp=None 
            ))

        return out
    def export_swap_log_to_csv(self, path: str, append: bool = False):
        if not hasattr(self, "swap_log") or not self.swap_log:
            print("No swap logs to export."); return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mode = "a" if append else "w"
        file_exists = os.path.exists(path)
        import csv
        with open(path, mode, newline="", encoding="utf-8") as f:
            fieldnames = list(asdict(self.swap_log[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not append or not file_exists:
                writer.writeheader()
            for rec in self.swap_log:
                writer.writerow(asdict(rec))

    def export_swap_log_to_json(self, path: str, append: bool = False):
        if not hasattr(self, "swap_log") or not self.swap_log:
            print("No swap logs to export."); return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            for rec in self.swap_log:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
    def _append_csv_row_locked(self, path: str, fieldnames: List[str], row: dict):
        import csv

        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.exists(path)

        with open(path, "a", newline="", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

            need_header = (not file_exists) or (f.tell() == 0)
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if need_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            f.flush()

            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def _append_jsonl_row_locked(self, path: str, row: dict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def _log_population_top10_after_update(self, generation: int):
        ts = time.time()
        oracle_name = self.args.oracles[0] if getattr(self, "args", None) and getattr(self.args, "oracles", None) else ""

        for i, pool in enumerate(self.pools):
            scores = list(getattr(pool, "scores", []) or [])
            mols = list(getattr(pool, "mols", []) or [])
            scores = list(getattr(pool, "scores", []) or [])

            top10_idx = self._select_topk_by_score_and_avg_tanimoto(
                mols=mols,
                scores=scores,
                k=10,
                sim_threshold=0.4,
                require_greater=False,  
                fallback_fill=True,
            )

            if top10_idx:
                top10 = float(np.mean([scores[j] for j in top10_idx]))
            else:
                top10 = float("nan")
            best  = float(max(scores)) if len(scores) > 0 else float("nan")
            size  = int(len(getattr(pool, "mols", []) or [])) 
            oracle_calls = int(len(self.oracle))
            row = {
                "timestamp": ts,
                "oracle": oracle_name,
                "seed": self.seed,
                "generation": int(generation),
                "global_step": int(getattr(self, "global_step", 0)),
                "oracle_calls": oracle_calls,  
                "pool_idx": int(i),
                "pool_inverse_temperature": float(getattr(pool, "inverse_temperature", float("nan"))),
                "pool_size": size,
                "top10_avg": top10,
                "best": best,
            }

            self._append_csv_row_locked(self.pop_csv_path, self.pop_csv_fieldnames, row)
            self._append_jsonl_row_locked(self.pop_jsonl_path, row)

    def _select_topk_by_score_and_avg_tanimoto(
        self,
        mols: List[Chem.Mol],
        scores: List[float],
        k: int = 10,
        sim_threshold: float = 0.4,
        require_greater: bool = True,  
        fp_radius: int = 2,
        fp_nbits: int = 2048,
        fallback_fill: bool = True,
    ) -> List[int]:
        if not mols or not scores:
            return []
        n = min(len(mols), len(scores))
        if n == 0:
            return []

        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)

        selected: List[int] = []
        selected_fps = []

        def _fp(m: Chem.Mol):
            if m is None:
                return None
            return AllChem.GetMorganFingerprintAsBitVect(m, fp_radius, nBits=fp_nbits)

        for idx in ranked:
            m = mols[idx]
            fp = _fp(m)
            if fp is None:
                continue

            if not selected:
                selected.append(idx)
                selected_fps.append(fp)
            else:
                sims = [DataStructs.TanimotoSimilarity(fp, fp2) for fp2 in selected_fps]
                avg_sim = float(sum(sims) / len(sims)) if sims else 0.0
                ok = (avg_sim > sim_threshold) if require_greater else (avg_sim < sim_threshold)
                if ok:
                    selected.append(idx)
                    selected_fps.append(fp)

            if len(selected) >= k:
                break

        if fallback_fill and len(selected) < k:
            selected_set = set(selected)
            for idx in ranked:
                if idx in selected_set:
                    continue
                m = mols[idx]
                fp = _fp(m)
                if fp is None:
                    continue
                selected.append(idx)
                selected_fps.append(fp)
                if len(selected) >= k:
                    break

        return selected

    def _dump_population_this_round(self, generation: int):
        if self.popdump_every <= 0 or (generation % self.popdump_every) != 0:
            return

        ts = time.time()
        oracle_name = self.args.oracles[0] if getattr(self, "args", None) and getattr(self.args, "oracles", None) else ""
        oracle_calls = int(len(self.oracle)) if hasattr(self, "oracle") else None

        if self.popdump_mode == "per_round":
            if self.popdump_gzip:
                import gzip
                path = os.path.join(self.popdump_dir, f"gen_{generation:06d}.jsonl.gz")
                opener = lambda p: gzip.open(p, "wt", encoding="utf-8")
            else:
                path = os.path.join(self.popdump_dir, f"gen_{generation:06d}.jsonl")
                opener = lambda p: open(p, "w", encoding="utf-8")

            with opener(path) as f:
                for pool_idx, pool in enumerate(self.pools):
                    mols = list(getattr(pool, "mols", []) or [])
                    scores = list(getattr(pool, "scores", []) or [])
                    temp = float(getattr(pool, "inverse_temperature", float("nan")))

                    n = min(len(mols), len(scores))
                    for j in range(n):
                        m = mols[j]
                        smi = Chem.MolToSmiles(m) if m is not None else None
                        rec = {
                            "timestamp": ts,
                            "oracle": oracle_name,
                            "seed": self.seed,
                            "generation": int(generation),
                            "global_step": int(getattr(self, "global_step", 0)),
                            "oracle_calls": oracle_calls,
                            "pool_idx": int(pool_idx),
                            "pool_inverse_temperature": temp,
                            "idx_in_pool": int(j),
                            "smiles": smi,
                            "score": float(scores[j]) if scores[j] is not None else None,
                        }
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return

        path = self.popdump_append_path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass

            for pool_idx, pool in enumerate(self.pools):
                mols = list(getattr(pool, "mols", []) or [])
                scores = list(getattr(pool, "scores", []) or [])
                temp = float(getattr(pool, "inverse_temperature", float("nan")))

                n = min(len(mols), len(scores))
                for j in range(n):
                    m = mols[j]
                    smi = Chem.MolToSmiles(m) if m is not None else None
                    rec = {
                        "timestamp": ts,
                        "oracle": oracle_name,
                        "seed": self.seed,
                        "generation": int(generation),
                        "global_step": int(getattr(self, "global_step", 0)),
                        "oracle_calls": oracle_calls,
                        "pool_idx": int(pool_idx),
                        "pool_inverse_temperature": temp,
                        "idx_in_pool": int(j),
                        "smiles": smi,
                        "score": float(scores[j]) if scores[j] is not None else None,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
