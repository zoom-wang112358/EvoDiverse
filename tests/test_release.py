"""Offline integration checks: no paid APIs and no benchmark downloads."""
import asyncio
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for directory in [ROOT, ROOT / 'molecule_optimization', ROOT / 'algo_discovery',
                  ROOT / 'equation_discovery/symbolicreg',
                  ROOT / 'equation_discovery/symbolicreg/methods']:
    sys.path.insert(0, str(directory))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(root):
    import h5py
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    (root / 'data').mkdir()
    metadata = [{'name': 'BPG0', 'symbols': ['y', 'x'],
                 'symbol_descs': ['synthetic output', 'synthetic input'],
                 'symbol_properties': ['O', 'V'], 'expression': '2*x**2+0.3'}]
    pq.write_table(pa.Table.from_pylist(metadata), root / 'data/lsr_synth_bio_pop_growth-00000-of-00001.parquet')
    with h5py.File(root / 'lsr_bench_data.hdf5', 'w') as handle:
        group = handle.create_group('/lsr_synth/bio_pop_growth/BPG0')
        for split, limit in [('train_data', 1.0), ('id_test_data', 0.9), ('ood_test_data', 1.5)]:
            x = np.linspace(0.1, limit, 16)
            group.create_dataset(split, data=np.column_stack([2*x*x+0.3, x]))


class ReleaseTests(unittest.TestCase):
    def test_help_without_credentials(self):
        for script in ['molecule_optimization/run.py', 'equation_discovery/symbolicreg/eval.py',
                       'algo_discovery/run.py']:
            result = subprocess.run([sys.executable, str(ROOT / script), '--help'],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--check', result.stdout)

    def test_release_file_boundaries(self):
        forbidden = {'.yaml', '.yml', '.hdf5', '.h5', '.parquet', '.pkl', '.csv', '.jsonl', '.zip'}
        for path in ROOT.rglob('*'):
            if '.git' in path.parts or '__pycache__' in path.parts:
                continue
            self.assertFalse(path.is_symlink(), str(path))
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), forbidden, str(path))

    def test_sr_log_mse_swap(self):
        import math
        from llmsr.pt_math_utils import compute_logA_from_rewards
        result = compute_logA_from_rewards(-0.01, -1.0, 0.8, 0.01, xi=2.5)
        self.assertAlmostEqual(result, 2.5*(0.8-0.01)*(math.log(0.01)-math.log(1.0)))
        self.assertLess(result, 0)
        self.assertAlmostEqual(result, -compute_logA_from_rewards(-1.0, -0.01, 0.8, 0.01, xi=2.5))

    def test_public_hf_data_reader_and_check(self):
        from bench.datamodules.synth import HFSynthDataModule
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_dataset(root)
            dm = HFSynthDataModule('bio_pop_growth', root)
            dm.setup()
            self.assertEqual(dm.problems[0].equation_idx, 'BPG0')
            self.assertEqual(dm.problems[0].train_samples.shape, (16, 2))
            result = subprocess.run([
                sys.executable, str(ROOT / 'equation_discovery/symbolicreg/eval.py'),
                '--dataset', 'bio_pop_growth', '--data-root', str(root), '--problem-name', 'BPG0', '--check',
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Loaded 1 problem', result.stdout)

    def test_sr_search_with_fake_completions(self):
        from bench.datamodules.synth import HFSynthDataModule
        from bench.pipelines.base import EvaluationPipeline
        from llmsr import config
        from llmsr.searcher import EvoDiverseSearcher
        calls = []

        class FakeLLM:
            def __init__(self, samples_per_prompt):
                pass

            async def async_draw_single_sample(self, prompt, config):
                calls.append(prompt)
                return '    return params[0] * x**2 + params[1]\n'

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            make_dataset(root)
            dm = HFSynthDataModule('bio_pop_growth', root)
            dm.setup()
            cfg = config.PTConfig(use_api=True, api_model='offline', samples_per_prompt=1)
            output = root / 'results'
            output.mkdir()
            searcher = EvoDiverseSearcher(
                name='offline', cfg=cfg, SamplerClass=FakeLLM, global_max_sample_num=2,
                log_path=str(output / 'search_logs'), pool_temperatures=(0.8, 0.01),
                migration_size=10, migration_interval=5, population_size=1000)
            EvaluationPipeline().evaluate_problems(dm.problems, searcher, output)
            records = [json.loads(line) for line in (output / 'results.jsonl').read_text().splitlines()]
            self.assertTrue(calls)
            self.assertEqual(len(records), 1)
            self.assertIn('x**2', records[0]['discovered_program'])
            self.assertIsNotNone(records[0]['id_metrics'])
            self.assertGreater(records[0]['best_program_score'], -1e-6)

    def test_sr_api_routing(self):
        from llmsr import config, sampler
        calls = []

        class FakeClient:
            def __init__(self, **kwargs):
                self.responses = SimpleNamespace(create=self.response)
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat_response))

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def response(self, **kwargs):
                calls.append(('responses', kwargs['model']))
                return SimpleNamespace(output_text='def equation(x, params):\n    return x\n')

            async def chat_response(self, **kwargs):
                calls.append(('chat', kwargs['model']))
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                    content='def equation(x, params):\n    return x\n'))])

        with patch.object(sampler, 'AsyncOpenAI', FakeClient):
            llm = sampler.APILanguageModel(1, api_url='http://offline.invalid/v1', api_key='test-placeholder')
            for model in ['DeepSeek-V3.2', 'gpt-5']:
                text = asyncio.run(llm.async_draw_single_sample('test', config.PTConfig(use_api=True, api_model=model)))
                self.assertIn('return x', text)
        self.assertEqual(calls, [('chat', 'DeepSeek-V3.2'), ('responses', 'gpt-5')])

    def test_molecule_oracle_buffers_are_independent(self):
        from evodiverse_molecule.base_optimizer import Oracle
        with patch('tdc.Oracle'), patch('tdc.Evaluator'):
            first, second = Oracle(), Oracle()
        first.mol_buffer['CC'] = [0.5, 1, 0.8]
        self.assertEqual(second.mol_buffer, {})
        self.assertIsNot(first.combined_buffer, second.combined_buffer)

    def test_molecule_api_failures_stop_search(self):
        import httpx
        from openai import AuthenticationError
        from rdkit import Chem
        from evodiverse_molecule import llm_proposer

        proposer = llm_proposer.LLMMoleculeProposer()
        proposer.task = ['jnk3']
        response = httpx.Response(401, request=httpx.Request('POST', 'https://offline.invalid/v1/chat/completions'))
        failure = AuthenticationError('Invalid API credentials', response=response, body=None)
        parents = [(0.5, Chem.MolFromSmiles('CCOc1ccccc1'), 1.0)]
        with patch.object(llm_proposer, 'query_llm', side_effect=failure) as query, \
             patch.object(llm_proposer.co, 'crossover') as crossover:
            with self.assertRaises(AuthenticationError):
                proposer.propose(parents, mutation_rate=0.067, temp=1.3, query_mode='cold')
        query.assert_called_once()
        crossover.assert_not_called()

    def test_molecule_genetic_fallback_for_invalid_completions(self):
        import random
        import numpy as np
        from rdkit import Chem
        from evodiverse_molecule import llm_proposer

        random_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            random.seed(0)
            np.random.seed(0)
            proposer = llm_proposer.LLMMoleculeProposer()
            proposer.task = ['jnk3']
            parents = [(0.5, Chem.MolFromSmiles('CCOc1ccccc1'), 1.0),
                       (0.6, Chem.MolFromSmiles('CCNCCc1ccccc1'), 1.0)]
            with patch.object(llm_proposer, 'query_llm', return_value=(None, 'Invalid completion')) as query, \
                 patch.object(llm_proposer.co, 'crossover', wraps=llm_proposer.co.crossover) as crossover, \
                 contextlib.redirect_stdout(io.StringIO()) as log:
                child, _ = proposer.propose(parents, mutation_rate=0.067, temp=1.3, query_mode='cold')
            self.assertEqual(query.call_count, 3)
            crossover.assert_called_once()
            self.assertIn('GA New child SMILES:', log.getvalue())
            self.assertNotIn('NameError', log.getvalue())
            self.assertIsNotNone(child)
            self.assertIsNotNone(Chem.MolFromSmiles(Chem.MolToSmiles(child)))
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)

    def test_molecule_search_with_fake_oracles_and_proposals(self):
        import itertools
        from rdkit import Chem
        from evodiverse_molecule.optimizer import EvoDiverseOptimizer
        from evodiverse_molecule.llm_proposer import LLMMoleculeProposer
        from evodiverse_runtime import working_directory
        import tdc.chem_utils.oracle.filter

        class FakeOracle:
            name = 'jnk3'
            def __call__(self, smiles):
                def score(smi):
                    return 0.7 + min(len(smi), 20) * 0.005
                return score(smiles) if isinstance(smiles, str) else [score(smi) for smi in smiles]

        children = itertools.count(5)
        def propose(*args, **kwargs):
            return Chem.MolFromSmiles('C' * next(children)), {}

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            smis = root / 'initial.smi'
            smis.write_text('CCO\nCCN\nCCC\nCCCl\n')
            args = SimpleNamespace(model='offline', n_jobs=1, smi_file=str(smis),
                                   oracles=['jnk3'], method='evodiverse', seed=0,
                                   output_dir=str(root), max_oracle_calls=24,
                                   freq_log=100, patience=5, log_results=False)
            config = dict(population_size=4, initial_population_size=4, offspring_size=2,
                          mutation_rate=0.067, importance_reweight=0.0)
            with working_directory(root), patch('tdc.Oracle', return_value=FakeOracle()), \
                 patch('tdc.Evaluator', return_value=lambda smiles: 0.5), \
                 patch('tdc.chem_utils.oracle.filter.MolFilter', return_value=lambda smiles: smiles), \
                 patch.object(LLMMoleculeProposer, 'propose', propose):
                optimizer = EvoDiverseOptimizer(args=args)
                optimizer.optimize(FakeOracle(), config, seed=0)
            self.assertTrue(list(root.glob('*results*.yaml')))
            self.assertTrue(list((root / 'population_dumps').rglob('*.json*')))
            self.assertTrue((root / 'logs/swap_log.csv').is_file())
            self.assertTrue((root / 'logs/swap_log.jsonl').is_file())
            swap_records = [json.loads(line) for line in
                            (root / 'logs/swap_log.jsonl').read_text().splitlines()]
            self.assertGreater(len(swap_records), 0)

    def test_sr_request_failure_does_not_hang(self):
        from llmsr import config
        from llmsr.pt_sampler import PTSampler

        class FailingLLM:
            def __init__(self, samples_per_prompt):
                pass
            async def async_draw_single_sample(self, prompt, cfg):
                raise RuntimeError('offline simulated request failure')

        prompt = SimpleNamespace(code='test', island_id=0, pool_id=0, version_generated=1)
        database = SimpleNamespace(get_prompt=lambda: prompt)
        sampler = PTSampler(database, [], 1, config.PTConfig(),
                            max_sample_nums=2, llm_class=FailingLLM)
        sampler.set_global_sample_nums(1)
        with self.assertRaisesRegex(RuntimeError, 'simulated request failure'):
            sampler.sample()

    def test_circle_preset_and_initial_program(self):
        runner = load_file('circle_runner', ROOT / 'algo_discovery/run.py')
        config = runner.make_config('offline', 'test-placeholder', 'http://offline.invalid/v1',
                                    1, 1, Path('/tmp/evodiverse-offline'))
        self.assertEqual(config.database.boltzmann_xi, 5.0)
        self.assertEqual(config.database.pool_temperatures, [1.0, 4.0])
        result = subprocess.run([sys.executable, str(ROOT / 'algo_discovery/run.py'), '--check'],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)


if __name__ == '__main__':
    unittest.main()
