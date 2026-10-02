# EvoDiverse

Code for **Towards Diverse Scientific Hypothesis Search with Large Language Models**.

EvoDiverse maintains cold and hot populations of scientific hypotheses and exchanges
candidates using parallel tempering. This repository provides the EvoDiverse search
method for molecular optimization, equation discovery, and circle packing.

| Task | Example | Instructions |
| --- | --- | --- |
| Molecular optimization | JNK3 and GSK3B | [molecule_optimization](molecule_optimization/README.md) |
| Equation discovery | Four LLM-SRBench scientific domains | [equation_discovery](equation_discovery/README.md) |
| Algorithm discovery | Packing 26 circles in a unit square | [algo_discovery](algo_discovery/README.md) |

Use **Python 3.11** on Linux. Each task has its own `requirements.txt`; a separate
virtual environment per task is recommended. Run the commands below from the
repository root. The modified OpenEvolve implementation is included locally.

## Model access

The runners use an OpenAI-compatible endpoint. Equation discovery also supports
GPT-5 through the Responses API. Configure your own provider's base URL, deployment
name, and API key:

```bash
cp .env.example .env
# Edit .env locally, then export its variables:
set -a
source .env
set +a
```

`EVODIVERSE_BASE_URL` is the API base URL (usually ending in `/v1`), not a chat
completion URL. Set `EVODIVERSE_MODEL` to the actual model/deployment name accepted
by that endpoint. A `--model` argument overrides the model environment variable.

## Minimal runs

Install the requirements for the task you want to use first. Each `--check` command
checks local inputs without calling a model API.

```bash
# Molecular discovery; PyTDC obtains its ZINC pool and oracle assets on a real run.
python molecule_optimization/run.py --check
python molecule_optimization/run.py --oracles jnk3 --seed 0 --max-oracle-calls 500

# Equation discovery; download the benchmark first, as described in its README.
python equation_discovery/symbolicreg/eval.py \
  --dataset bio_pop_growth --data-root /path/to/llm-srbench --problem-name BPG0 --check
python equation_discovery/symbolicreg/eval.py \
  --dataset bio_pop_growth --data-root /path/to/llm-srbench --problem-name BPG0 \
  --max-samples 16

# Circle packing; no task data download is required.
python algo_discovery/run.py --check
python algo_discovery/run.py --iterations 5 --seed 1
```

These reduced budgets are installation examples. The task READMEs give the main
experiment budgets. Runs create timestamped subdirectories under `outputs/` by
default. Data, credentials, and generated outputs are ignored by Git.

Generated Python programs are executed locally for equation and algorithm search;
run these tasks in an isolated environment suitable for executing generated code.

## Search settings and tests

See [tests/README.md](tests/README.md) for tests that run without model API calls.

## Citation
If you find our work helpful, please consider citing our paper:

```
@article{
      wang2026towards,
      title={Towards Diverse Scientific Hypothesis Search with Large Language Models},
      author={Wang, Haorui and Shojaee, Parshin and Meidani, Kazem and Sun, Kunyang and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Head-Gordon, Teresa and He, Jiajun and Reddy, Chandan K and Zhang, Chao and Du, Yuanqi},
      booktitle={ICML},
      year={2026},
      url={https://arxiv.org/html/2606.10587v1}
}
```
