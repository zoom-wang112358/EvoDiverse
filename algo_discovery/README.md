# Circle packing

This task evolves Python programs that pack 26 circles inside a unit square,
maximizing the sum of radii. EvoDiverse builds on
[OpenEvolve](https://github.com/codelion/openevolve).

From the repository root, using Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r algo_discovery/requirements.txt
python algo_discovery/run.py --check
```

The check evaluates the included initial program without calling an LLM. No task
data download is needed. Configure your model endpoint following the root README,
then run the main experiment budget:

```bash
python algo_discovery/run.py --iterations 1000 --seed 1
```

Use `--seed` to select a random seed for independent runs. The default is 42.

The Python preset in `run.py` uses two pools with `T = [1.0, 4.0]`, swaps every
five iterations, initial `xi = 5.0`, target swap acceptance 0.3, and a mating pool
of 20 programs. The initial program and evaluator are under
`examples/circle_packing/`; the PT selection and swap logic is in
`openevolve/database.py`.

Run `python algo_discovery/run.py --help` for output and model options. Every run
gets a separate directory under `outputs/circle_packing/`.
