import os
import yaml
import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import tdc
from tdc.generation import MolGen
from .chemistry import *
import math
import json


class Objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)


def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls, temp=None, temp_tol=1e-8):
    """
    buffer: dict {smi: [score, call_idx, temp]  or  [score, call_idx]}
    temp:   None => global, float or str => filter by temp
    """

    def temp_match(v):
        # v = buffer[smi] = [score, call_idx, temp?]
        if temp is None:
            return True
        if len(v) < 3:
            return False
        t = v[2]
        if isinstance(temp, (float, np.floating)) and isinstance(t, (float, np.floating)):
            return abs(float(t) - float(temp)) <= temp_tol
        return t == temp

    ordered_all = sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False)

    total_calls = min(len(ordered_all), max_oracle_calls)
    if total_calls == 0:
        return 0.0

    area = 0.0
    prev = 0.0
    called = 0

    for idx in range(freq_log, total_calls, freq_log):
        prefix = ordered_all[:idx]
        prefix = [kv for kv in prefix if temp_match(kv[1])]

        if len(prefix) == 0:
            top_n_now = 0.0
        else:
            best = sorted(prefix, key=lambda kv: kv[1][0], reverse=True)[:top_n]
            top_n_now = float(np.mean([item[1][0] for item in best]))

        area += freq_log * (top_n_now + prev) / 2.0
        prev = top_n_now
        called = idx

    prefix = ordered_all[:total_calls]
    prefix = [kv for kv in prefix if temp_match(kv[1])]

    if len(prefix) == 0:
        top_n_now = 0.0
    else:
        best = sorted(prefix, key=lambda kv: kv[1][0], reverse=True)[:top_n]
        top_n_now = float(np.mean([item[1][0] for item in best]))

    area += (total_calls - called) * (top_n_now + prev) / 2.0

    if finish and total_calls < max_oracle_calls:
        area += (max_oracle_calls - total_calls) * top_n_now

    return area / float(max_oracle_calls)


class Oracle:
    def __init__(self, args=None, mol_buffer=None, combined_buffer=None):
        self.name = None
        self.evaluator = None
        self.task_label = None
        if args is None:
            self.max_oracle_calls = 10000
            self.freq_log = 100
        else:
            self.args = args
            self.max_oracle_calls = args.max_oracle_calls
            self.freq_log = args.freq_log

        self.mol_buffer = {} if mol_buffer is None else mol_buffer
        self.combined_buffer = {} if combined_buffer is None else combined_buffer
        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.last_log = 0

        self.oracle_name=None


    @property
    def budget(self):
        return self.max_oracle_calls

    def assign_evaluator(self, evaluator):
        self.evaluator = evaluator

    def sort_buffer(self):
        self.mol_buffer = dict(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))
    def save_result(self, suffix=None):

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, self.args.method + '_' + 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, self.args.method + '_' + 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)


    def log_intermediate(self, finish=False, temp=None, temp_tol=1e-8):
        """
        """

        if len(self.mol_buffer) == 0:
            print("No data to log.")
            return

        ordered_all = sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False)
        n_calls = min(len(ordered_all), self.max_oracle_calls)
        prefix_all = ordered_all[:n_calls]

        def temp_match(v):
            if temp is None:
                return True
            if len(v) < 3:
                return False
            t = v[2]
            if isinstance(temp, (float, np.floating)) and isinstance(t, (float, np.floating)):
                return abs(float(t) - float(temp)) <= temp_tol
            return t == temp

        items = [kv for kv in prefix_all if temp_match(kv[1])]

        if not items:
            print(f"No data to log for temp={temp} within first {n_calls} calls.")
            return

        items_by_score = sorted(items, key=lambda kv: kv[1][0], reverse=True)
        top100 = items_by_score[:100]
        smis = [x[0] for x in top100]
        scores_top100 = [x[1][0] for x in top100]

        avg_top1 = float(scores_top100[0])
        avg_top10 = float(np.mean(scores_top100[:10])) if len(scores_top100) >= 10 else float(np.mean(scores_top100))
        avg_top100 = float(np.mean(scores_top100))

        avg_sa = float(np.mean(self.sa_scorer(smis))) if smis else 0.0
        diversity_top100 = float(self.diversity_evaluator(smis)) if smis else 0.0

        auc1 = top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls, temp=temp)
        auc10 = top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls, temp=temp)
        auc100 = top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls, temp=temp)

        print(f'[GLOBAL] temp={temp} | active={len(items)} | {n_calls}/{self.max_oracle_calls} | '
            f'avg_top1: {avg_top1:.3f} | avg_top10: {avg_top10:.3f} | avg_top100: {avg_top100:.3f} | '
            f'avg_sa: {avg_sa:.3f} | div: {diversity_top100:.3f}')

        print({
            "temp": temp,
            "active_size": len(items),
            "avg_top1": avg_top1,
            "avg_top10": avg_top10,
            "avg_top100": avg_top100,
            "auc_top1": auc1,
            "auc_top10": auc10,
            "auc_top100": auc100,
            "avg_sa": avg_sa,
            "diversity_top100": diversity_top100,
            "n_oracle_global": n_calls,
        })
        if finish:
            n_calls = self.max_oracle_calls
        if n_calls == self.max_oracle_calls:
            finish = True
            from pathlib import Path
            out_dir = Path.cwd() / 'metrics' / 'summary'
            out_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{self.args.oracles[0]}_{self.args.method}.txt"
            output_path = out_dir / filename

            result_list = [
                temp, avg_top1, avg_top10, avg_top100,
                top_auc(self.mol_buffer, 1, finish, self.freq_log, self.max_oracle_calls, temp=temp),
                top_auc(self.mol_buffer, 10, finish, self.freq_log, self.max_oracle_calls, temp=temp),
                top_auc(self.mol_buffer, 100, finish, self.freq_log, self.max_oracle_calls, temp=temp),
                avg_sa, diversity_top100, n_calls
            ]
            with output_path.open('a', encoding='utf-8') as f:
                f.write(','.join(map(str, result_list)) + '\n')


    def __len__(self):
        return len(self.mol_buffer)

    def score_smi(self, smi, temp):
        """
        Function to score one molecule (single)
        """
        if len(self.mol_buffer) >= self.max_oracle_calls:
            return 0
        if smi is None:
            return 0

        mol = Chem.MolFromSmiles(smi)
        if mol is None or len(smi) == 0:
            return 0

        smi = Chem.MolToSmiles(mol)  

        if smi not in self.mol_buffer:
            fitness = float(self.evaluator(smi))  
            if math.isnan(fitness):
                fitness = 0

            self.mol_buffer[smi] = [fitness, len(self.mol_buffer) + 1, temp]

        return self.mol_buffer[smi][0]

    def __call__(self, smiles_lst, temp=0.8):
        """
        Score a batch of molecules and cache the oracle results.

        ``temp`` labels the source pool in the cache. The evaluator receives
        a list of SMILES for batch scoring.
        """
        temp_list = [0.2, 0.8]


        is_single = isinstance(smiles_lst, str)
        if is_single:
            smiles_in = [smiles_lst]
        else:
            smiles_in = list(smiles_lst)  

        scores = [0.0] * len(smiles_in)

        to_eval = []
        to_eval_idx = []
        to_eval_smi = []


        remaining = max(0, self.max_oracle_calls - len(self.mol_buffer))

        for i, smi in enumerate(smiles_in):
            if smi is None or len(str(smi)) == 0:
                scores[i] = 0.0
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                scores[i] = 0.0
                continue

            can_smi = Chem.MolToSmiles(mol)

            if can_smi in self.mol_buffer:
                scores[i] = self.mol_buffer[can_smi][0]
                continue

            if remaining <= 0:
                scores[i] = 0.0
                continue

            to_eval.append(can_smi)
            to_eval_idx.append(i)
            to_eval_smi.append(can_smi)
            remaining -= 1

        if to_eval:
            fitness_list = self.evaluator(to_eval)

            try:
                fitness_iter = list(fitness_list)
            except TypeError:
                fitness_iter = [fitness_list]

            for can_smi, idx, fit in zip(to_eval_smi, to_eval_idx, fitness_iter):
                try:
                    fitness = float(fit)
                except Exception:
                    fitness = 0.0

                if math.isnan(fitness):
                    fitness = 0.0

                self.mol_buffer[can_smi] = [fitness, len(self.mol_buffer) + 1, temp]
                scores[idx] = fitness

        if len(self.mol_buffer) > self.last_log:
            self.sort_buffer()
            for t in temp_list:
                self.log_intermediate(temp=t)
            self.last_log = len(self.mol_buffer)
            self.save_result(self.task_label)

        return scores[0] if is_single else scores

    @property
    def finish(self):
        return len(self.mol_buffer) >= self.max_oracle_calls


class BaseOptimizer:

    @staticmethod
    def load_smiles_from_file(path):
        with open(path, encoding="utf-8") as handle:
            smiles = [line.split()[0] for line in handle
                      if line.strip() and not line.lstrip().startswith("#")]
        if not smiles:
            raise ValueError("The initial SMILES file is empty.")
        return smiles

    def __init__(self, args=None):
        self.model_name = args.model
        self.args = args
        self.n_jobs = args.n_jobs
        self.smi_file = args.smi_file
        self.oracle = Oracle(args=self.args)
        if self.smi_file is not None:
            self.all_smiles = self.load_smiles_from_file(self.smi_file)
        else:
            data = MolGen(name = 'ZINC')
            self.all_smiles = data.get_data()['smiles'].tolist()

        self.sa_scorer = tdc.Oracle(name = 'SA')
        self.diversity_evaluator = tdc.Evaluator(name = 'Diversity')
        self.filter = tdc.chem_utils.oracle.filter.MolFilter(filters = ['PAINS', 'SureChEMBL', 'Glaxo'], property_filters_flag = False)

    def sanitize(self, mol_list):
        new_mol_list = []
        smiles_set = set()

        for mol in mol_list:
            if mol is None:
                continue

            if isinstance(mol, (tuple, list)):
                if len(mol) == 0:
                    continue
                mol = mol[0]

            if not isinstance(mol, Chem.Mol):
                continue

            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles and smiles not in smiles_set:
                    smiles_set.add(smiles)
                    new_mol_list.append(mol)
            except Exception as e:
                print("bad mol:", e)
                continue

        return new_mol_list

    def sort_buffer(self):
        self.oracle.sort_buffer()

    def log_intermediate(self, mols=None, scores=None, finish=False):
        self.oracle.log_intermediate(finish=finish, temp=0.2)
        self.oracle.log_intermediate(finish=finish, temp=0.8)

        from pathlib import Path
        out_dir = Path.cwd() / 'metrics' / 'per_pool'
        out_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.args.oracles[0]}_{self.args.method}.txt"
        output_path = out_dir / filename
        summary = self.report_all_pools_metrics(topk=10, div_k=100, auc_freq_log=50)
        print("[FINAL PER-POOL REPORT]", summary)
        with open(output_path, "a", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")

    def log_result(self):

        print(f"Logging final results...")


        log_num_oracles = [100, 500, 1000, 3000, 5000, 10000]
        assert len(self.mol_buffer) > 0

        results = list(sorted(self.mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
        if len(results) > 10000:
            results = results[:10000]

        results_all_level = []
        for n_o in log_num_oracles:
            results_all_level.append(sorted(results[:n_o], key=lambda kv: kv[1][0], reverse=True))


        # Log batch metrics at various oracle calls
        data = [[log_num_oracles[i]] + self._analyze_results(r) for i, r in enumerate(results_all_level)]
        columns = ["#Oracle", "avg_top100", "avg_top10", "avg_top1", "Diversity", "avg_SA", "%Pass", "Top-1 Pass"]

    def save_result(self, suffix=None):

        print(f"Saving molecules...")

        if suffix is None:
            output_file_path = os.path.join(self.args.output_dir, self.args.method + '_' + 'results.yaml')
        else:
            output_file_path = os.path.join(self.args.output_dir, self.args.method + '_' + 'results_' + suffix + '.yaml')

        self.sort_buffer()
        with open(output_file_path, 'w') as f:
            yaml.dump(self.mol_buffer, f, sort_keys=False)

    def _analyze_results(self, results):
        results = results[:100]
        scores_dict = {item[0]: item[1][0] for item in results}
        smis = [item[0] for item in results]
        scores = [item[1][0] for item in results]
        smis_pass = self.filter(smis)
        if len(smis_pass) == 0:
            top1_pass = -1
        else:
            top1_pass = np.max([scores_dict[s] for s in smis_pass])
        return [np.mean(scores),
                np.mean(scores[:10]),
                np.max(scores),
                self.diversity_evaluator(smis),
                np.mean(self.sa_scorer(smis)),
                float(len(smis_pass) / 100),
                top1_pass]

    def reset(self):
        try:
            super().reset()
        except Exception:
            pass

        try:
            del self.oracle
        except Exception:
            pass
        self.oracle = Oracle(args=self.args)

        from collections import defaultdict, deque
        self.pools = []
        self.swap_stats = defaultdict(lambda: {"proposals": 0, "accepts": 0})
        self.swap_hist = defaultdict(lambda: deque(maxlen=200))
        self.swap_stats_global = {"proposals": 0, "accepts": 0}
        self.swap_hist_global = deque(maxlen=200)
        self.swap_log = []
        self.global_step = 0
        self.migration_step = 0
        self.metrics_log = []

    @property
    def mol_buffer(self):
        return self.oracle.mol_buffer

    @property
    def finish(self):
        return self.oracle.finish

    def _optimize(self, oracle, config):
        raise NotImplementedError


    def report_all_pools_metrics(
        self,
        topk: int = 10,
        div_k: int = 100,
        auc_freq_log: int = 50,
        auc_max_calls: Optional[int] = None,
        finish: bool = True,
        temp_tol: float = 1e-8,
    ):
        pass
    def optimize(self, oracle, config, seed=0):

        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        self.oracle.task_label = self.args.model + "_" + oracle.name + "_" + str(seed)
        self._optimize(oracle, config)
        if self.args.log_results:
            self.log_result()
        self.save_result(self.args.model + "_" + oracle.name + "_" + str(seed))
        self.reset()
