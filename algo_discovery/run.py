"""Run EvoDiverse on circle packing with 26 circles."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from evodiverse_runtime import add_api_arguments, configure_api, positive_int


def build_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=positive_int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'circle_packing')
    add_api_arguments(parser)
    return parser


def make_config(model, key, base_url, iterations, seed, output):
    from openevolve.config import Config
    return Config.from_dict({
        'llm': {'api_base': base_url, 'api_key': key,
                'models': [{'name': model}], 'evaluator_models': [{'name': model}],
                'max_tokens': 4096, 'explorer_llm_temp': 1.0, 'explorer_llm_top_p': 0.95,
                'exploiter_llm_temp': 0.7, 'exploiter_llm_top_p': 0.8},
        'database': {'num_islands': 2, 'pool_temperatures': [1.0, 4.0],
                     'use_map_elites': False, 'use_tempering_selection': True,
                     'migration_interval': 5, 'migration_swap_rate': 1.0,
                     'boltzmann_xi': 5.0, 'target_swap_rate': 0.3,
                     'archive_size': 1000, 'feature_dimensions': ['score'], 'feature_bins': 10,
                     'mating_pool_size': 20, 'num_parents_per_offspring': 3},
        'evaluator': {'parallel_evaluations': 4, 'timeout': 300},
        'max_iterations': iterations, 'checkpoint_interval': 50,
        'random_seed': seed, 'log_level': 'INFO',
        'evolution_trace': {'enabled': True, 'format': 'jsonl', 'include_code': True,
                            'include_prompts': False, 'output_path': str(output / 'trace.jsonl')},
    })


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from openevolve.api import run_evolution
    example = HERE / 'examples' / 'circle_packing'
    if args.check:
        import importlib.util
        spec = importlib.util.spec_from_file_location('packing_evaluator', example / 'evaluator.py')
        evaluator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluator)
        metrics = evaluator.evaluate(str(example / 'initial_program.py'))
        values = getattr(metrics, 'metrics', metrics)
        if not isinstance(values, dict) or values.get('combined_score', 0) <= 0:
            parser.error(f'Initial packing validation failed: {values}')
        print(f'Circle packing imports and initial program OK: {values}')
        return
    key, base_url = configure_api(parser, args.model)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = args.output_dir.resolve() / f'seed_{args.seed}_{stamp}'
    cfg = make_config(args.model, key, base_url, args.iterations, args.seed, output)
    result = run_evolution(example / 'initial_program.py', example / 'evaluator.py',
                           config=cfg, iterations=args.iterations, output_dir=str(output))
    print(f'Best score: {result.best_score}; results: {output}')


if __name__ == '__main__':
    main()
