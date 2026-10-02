# Equation discovery

EvoDiverse searches equation programs using two pools with inverse temperatures
`beta = [0.8, 0.01]`. The swap acceptance rule uses log-MSE.

## Install

From the repository root, using Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r equation_discovery/requirements.txt
```

## Download the benchmark

Obtain the data from the original **[LLM-SRBench Hugging Face dataset](https://huggingface.co/datasets/nnheui/llm-srbench)**.
The [benchmark repository](https://github.com/deep-symbolic-mathematics/llm-srbench)
documents the scientific tasks. No task data is included in this repository.

The Hugging Face dataset requires access approval: sign in and accept the access
conditions on its dataset page before downloading the files.

Download `lsr_bench_data.hdf5` and the relevant `data/lsr_synth_*.parquet` files from
the dataset's **Files and versions** tab. Keep the following layout:

```text
/path/to/llm-srbench/
├── lsr_bench_data.hdf5
└── data/
    ├── lsr_synth_bio_pop_growth-00000-of-00001.parquet
    ├── lsr_synth_chem_react-00000-of-00001.parquet
    ├── lsr_synth_matsci-00000-of-00001.parquet
    └── lsr_synth_phys_osc-00000-of-00001.parquet
```

You only need the metadata file for a domain you plan to run, plus the shared HDF5
file. The loader reads these local files directly.

| `--dataset` | Scientific domain | Example problem ID |
| --- | --- | --- |
| `bio_pop_growth` | Biology | `BPG0` |
| `chem_react` | Chemistry | `CRK0` |
| `matsci` | Materials science | `MatSci0` |
| `phys_osc` | Physics | `PO0` |

## Run

Check the installation and downloaded data first:

```bash
python equation_discovery/symbolicreg/eval.py --dataset bio_pop_growth \
  --data-root /path/to/llm-srbench --problem-name BPG0 --check
```

Configure the endpoint in your environment as described in the root README. The
main experiment setting uses 1,000 samples per problem and eight samples per prompt:

```bash
python equation_discovery/symbolicreg/eval.py --dataset bio_pop_growth \
  --data-root /path/to/llm-srbench --max-samples 1000 --seed 0
```

Omit `--problem-name` to run the entire domain; add it for a single problem. Change
`--dataset` to run another domain. Use a DeepSeek-V3.2 deployment for that backbone,
or `--model gpt-5` with a compatible Responses API endpoint for GPT-5. Endpoint
configuration determines the available models.

The sampling loop checks the limit between batches, so the last batch
may finish beyond the requested sample threshold. Exact matches can stop early.
Outputs include the discovered program, ID/OOD evaluation metrics, and search
logs beneath `outputs/equation/`. The main entry point is `symbolicreg/eval.py`;
`EvoDiverseSearcher` in `symbolicreg/methods/llmsr/searcher.py` runs the search,
and `APILanguageModel` in `sampler.py` handles model requests. Population and
swap logic is under `symbolicreg/methods/llmsr/pt_*.py`.
