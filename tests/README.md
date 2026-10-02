# Offline validation

Install the three task requirement files into a Python 3.11 environment, then run
from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
PYTHONPATH=algo_discovery PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s algo_discovery/tests -v
```

The tests cover CLI loading, file boundaries, the log-MSE swap,
independent molecular oracle buffers, API failure propagation and genetic
fallback for invalid molecular completions, the circle-packing preset and evaluator,
the public Parquet/HDF5 data layout, API routing, and a small SR search using fake
model completions. Synthetic test data is created in a temporary directory.
No API key, benchmark download, or paid model request is used.

These checks validate local execution. Reproducing the paper's experiments
requires the task data, model access, and experiment budgets in the task READMEs.
