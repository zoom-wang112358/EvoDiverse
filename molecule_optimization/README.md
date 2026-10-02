# Molecular optimization

Run EvoDiverse on the PyTDC **JNK3** and **GSK3B** objectives with an LLM molecular
proposal operator. The main search uses two pools with inverse temperatures
`beta = [0.8, 0.2]`, population size 120 per pool, and 70 offspring per pool.

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r molecule_optimization/requirements.txt
python molecule_optimization/run.py --check
```

Use the dependency versions in `requirements.txt` for PyTDC/RDKit compatibility.
Git is required to install `rd_filters`. Use the included `rdkit-pypi` wheel as
the RDKit installation; installing a separate `rdkit` package can overwrite it.
The classical molecular oracle uses scikit-learn; no neural-model training or GPU
is required by this runner.

Configure model access following the root README, then run:

```bash
python molecule_optimization/run.py --oracles jnk3 gsk3b \
  --seed 0 1 2 --max-oracle-calls 10000
```

Each task and seed has a separate output directory containing run parameters and
results. The total oracle-call budget is shared across the two pools.

No dataset is distributed here. On a real run, **PyTDC downloads the ZINC initial
molecule pool and objective/scoring assets**. Those downloads require network
access and stay inside the run directory. You can supply an existing local SMILES
file with `--smi-file`; each nonempty line starts with one SMILES string.

Useful options: `--population-size`, `--offspring-size`, `--n-jobs`, `--model`, and
`--output-dir`. Use `python molecule_optimization/run.py --help` for details.

## Code structure

The `evodiverse_molecule` package contains:

- `optimizer.py`: the `EvoDiverseOptimizer` search loop.
- `llm_proposer.py`: the `LLMMoleculeProposer`, molecular prompts, and model requests.
- `base_optimizer.py`: oracle budgets, scoring, and result summaries.
- `core/`: population selection, diversity, and parallel-tempering swaps.
- `crossover.py` and `mutation.py`: genetic proposal operators.
- `chemistry.py` and `molecule_utils.py`: SMILES and molecular utilities.
- `logging_utils.py`: structured search logs.

Set `--model` to select your model deployment. Population inverse temperatures
are named `inverse_temperature` in the implementation; LLM sampling temperature
is a separate parameter.
