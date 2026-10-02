"""Run EvoDiverse on the JNK3 and GSK3B molecular objectives."""
from argparse import ArgumentParser, Namespace
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evodiverse_runtime import add_api_arguments, configure_api, positive_int, working_directory

# Default search hyperparameters.
DEFAULTS = dict(population_size=120, initial_population_size=120, offspring_size=70,
                mutation_rate=0.067, importance_reweight=0.0)


def build_parser():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--oracles', nargs='+', choices=['jnk3', 'gsk3b'], default=['jnk3'])
    parser.add_argument('--seed', nargs='+', type=int, default=[0])
    parser.add_argument('--max-oracle-calls', type=positive_int, default=10000)
    parser.add_argument('--population-size', type=positive_int, default=120)
    parser.add_argument('--offspring-size', type=positive_int, default=70)
    parser.add_argument('--n-jobs', type=positive_int, default=8)
    parser.add_argument('--smi-file', type=Path, help='Optional local initial SMILES file; otherwise use PyTDC ZINC.')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'molecule')
    add_api_arguments(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    from tdc import Oracle
    from tdc.chem_utils.oracle.filter import MolFilter
    from rd_filters.rd_filters import RDFilters, read_rules
    from evodiverse_molecule.optimizer import EvoDiverseOptimizer

    if args.smi_file is not None and not args.smi_file.is_file():
        parser.error('--smi-file does not exist.')
    if args.check:
        print('Molecular EvoDiverse imports OK. No oracle data or model API was requested.')
        return
    configure_api(parser, args.model)
    smi_file = str(args.smi_file.resolve()) if args.smi_file else None
    config = {**DEFAULTS, 'population_size': args.population_size,
              'initial_population_size': args.population_size, 'offspring_size': args.offspring_size}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output_root = args.output_dir.resolve()
    for oracle_name in args.oracles:
        for seed in args.seed:
            run_dir = output_root / oracle_name / f'seed_{seed}_{stamp}'
            run_dir.mkdir(parents=True, exist_ok=False)
            run_args = Namespace(method='evodiverse', model=args.model, oracles=[oracle_name],
                                 n_jobs=args.n_jobs, smi_file=smi_file, output_dir=str(run_dir),
                                 max_oracle_calls=args.max_oracle_calls, freq_log=100,
                                 patience=5, log_results=False, seed=seed,
                                 rate_dir=str(run_dir / 'rates'))
            metadata = {'task': oracle_name, 'seed': seed, 'model': args.model,
                        'max_oracle_calls': args.max_oracle_calls, 'hyperparameters': config}
            (run_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
            # Keep task outputs and PyTDC caches inside the run directory.
            with working_directory(run_dir):
                optimizer = EvoDiverseOptimizer(args=run_args)
                optimizer.optimize(oracle=Oracle(name=oracle_name), config=config, seed=seed)
            print(f'Results: {run_dir}')


if __name__ == '__main__':
    main()
