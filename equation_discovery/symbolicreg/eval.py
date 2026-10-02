"""Run EvoDiverse equation discovery on locally downloaded LLM-SRBench data."""
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import random
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / 'methods'))
from evodiverse_runtime import add_api_arguments, configure_api, positive_int


def build_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True,
                        choices=['matsci', 'chem_react', 'phys_osc', 'bio_pop_growth'])
    parser.add_argument('--data-root', required=True, type=Path,
                        help='Local Hugging Face download directory containing data/ and lsr_bench_data.hdf5.')
    parser.add_argument('--problem-name', help='Optional problem ID, e.g. BPG0. Default: all problems in the domain.')
    parser.add_argument('--max-samples', type=positive_int, default=1000)
    parser.add_argument('--samples-per-prompt', type=positive_int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'equation')
    add_api_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    import numpy as np
    from bench.datamodules.synth import HFSynthDataModule
    from bench.pipelines.base import EvaluationPipeline
    from llmsr import config, sampler
    from llmsr.searcher import EvoDiverseSearcher

    random.seed(args.seed)
    np.random.seed(args.seed)
    dm = HFSynthDataModule(args.dataset, args.data_root)
    try:
        dm.setup()
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    problems = dm.problems
    if args.problem_name:
        problems = [p for p in problems if p.equation_idx == args.problem_name]
        if not problems:
            parser.error(f'Unknown problem ID for {args.dataset}: {args.problem_name}')
    if args.check:
        print(f'Equation EvoDiverse imports OK. Loaded {len(problems)} problem(s); no model API was requested.')
        return
    key, base_url = configure_api(parser, args.model)
    cfg = config.PTConfig(experience_buffer=config.PTExperienceBufferConfig(), use_api=True,
                          api_model=args.model, samples_per_prompt=args.samples_per_prompt)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.output_dir.resolve() / args.dataset / f'seed_{args.seed}_{stamp}'
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'dataset': args.dataset, 'problem_ids': [p.equation_idx for p in problems],
                'seed': args.seed, 'max_samples': args.max_samples, 'config': asdict(cfg)}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    searcher = EvoDiverseSearcher(
        name='EvoDiverse', cfg=cfg,
        SamplerClass=lambda samples_per_prompt: sampler.APILanguageModel(
            samples_per_prompt=samples_per_prompt, api_url=base_url, api_key=key),
        global_max_sample_num=args.max_samples, log_path=str(output / 'search_logs'),
        pool_temperatures=cfg.experience_buffer.pool_temperatures,
        migration_size=cfg.experience_buffer.migration_size,
        migration_interval=cfg.experience_buffer.migration_interval,
        population_size=cfg.experience_buffer.population_size)
    EvaluationPipeline().evaluate_problems(problems, searcher, output)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
