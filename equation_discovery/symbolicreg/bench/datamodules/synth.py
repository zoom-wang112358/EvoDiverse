"""Read a locally downloaded copy of the public LLM-SRBench dataset.

Data source: https://huggingface.co/datasets/nnheui/llm-srbench
Layout follows the upstream benchmark's Parquet metadata and HDF5 samples.
"""
from pathlib import Path
import h5py
import numpy as np
import pyarrow.parquet as pq
from ..dataclasses import Equation, Problem

DATASET_URL = 'https://huggingface.co/datasets/nnheui/llm-srbench'
DOMAINS = ('matsci', 'chem_react', 'phys_osc', 'bio_pop_growth')


class SynProblem(Problem):
    @property
    def train_samples(self):
        return self.samples['train_data']

    @property
    def test_samples(self):
        return self.samples['id_test_data']

    @property
    def ood_test_samples(self):
        return self.samples['ood_test_data']


class HFSynthDataModule:
    def __init__(self, domain, root):
        if domain not in DOMAINS:
            raise ValueError(f'Unknown domain: {domain}')
        self.domain = domain
        self.root = Path(root)
        self.problems = []

    @property
    def name(self):
        return self.domain

    def setup(self):
        metadata_paths = sorted((self.root / 'data').glob(f'lsr_synth_{self.domain}-*.parquet'))
        samples_path = self.root / 'lsr_bench_data.hdf5'
        if not samples_path.is_file() or not metadata_paths:
            raise FileNotFoundError(
                f'Download lsr_bench_data.hdf5 and data/lsr_synth_{self.domain}-*.parquet '
                f'from {DATASET_URL} into {self.root}. See equation_discovery/README.md.'
            )
        problems = []
        with h5py.File(samples_path, 'r') as sample_file:
            for metadata_path in metadata_paths:
                for item in pq.read_table(metadata_path).to_pylist():
                    group = sample_file[f'/lsr_synth/{self.domain}/{item["name"]}']
                    samples = {key: np.asarray(group[key], dtype=np.float64)
                               for key in ('train_data', 'id_test_data', 'ood_test_data')}
                    for key, values in samples.items():
                        if values.ndim != 2 or values.shape[1] != len(item['symbols']):
                            raise ValueError(f'Unexpected shape in {item["name"]}/{key}: {values.shape}')
                    problems.append(SynProblem(
                        dataset_identifier=self.domain, equation_idx=item['name'],
                        gt_equation=Equation(symbols=item['symbols'], symbol_descs=item['symbol_descs'],
                                             symbol_properties=item['symbol_properties'],
                                             expression=item['expression']), samples=samples))
        self.problems = problems
        self.name2id = {p.equation_idx: i for i, p in enumerate(problems)}
