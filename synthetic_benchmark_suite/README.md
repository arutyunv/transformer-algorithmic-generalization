# benchmark_generator

Synthetic algorithmic-task generator for "How Transformer Architecture
Design Choices Affects Generalization on Algorithmic Tasks".

Implements the six benchmarks from the paper (Sec. 2.1 / Table 2):
`addition`, `sorting`, `dyck`, `kv`, `indexing`, `func` — each with
`train / val / test / test_ood` regimes, where `test_ood` shifts the
task-specific difficulty axis (digits, length, depth, n, ...) per the
paper's exact numbers.

## Tree

synthetic_benchmark_suite/
├── benchmark_generator/           — the main code package
│   ├── __init__.py                — exports the public API (Task, Split, get_task, BenchmarkDataset, etc.) for easy one-line imports
│   ├── base.py                    — foundation: Split (train/val/test/test_ood), Example (single prompt→target pair), and the abstract Task class that every benchmark must implement
│   ├── dataset.py                 — BenchmarkDataset wrapper: provides train as an infinite on-the-fly stream and val/test/test_ood as fixed, seed-reproducible sets
│   ├── registry.py                — "task name" → task class dictionary (get_task("addition"), etc.) to select tasks via strings
│   ├── cli.py                     — CLI utility to preview examples or dump them to .jsonl (python -m benchmark_generator.cli ...)
│   └── tasks/                     — the six data generators themselves
│       ├── __init__.py            — re-exports all task classes and their configs
│       ├── addition.py            — ADD task: number addition
│       ├── sorting.py             — SORT task: array sorting
│       ├── dyck.py                — DYCK task: balanced bracket sequence completion
│       ├── keyvalue.py            — KV task: value retrieval by key from a set of key:value pairs
│       ├── indexing.py            — INDEX task: retrieve array element by index
│       └── composition.py         — FUNC task: compute the composition of a chain of symbolic functions
│
├── README.md                      — instructions for installation and usage (imports, CLI)
└── pyproject.toml                 — metadata for `pip install -e .`, allows installing the package locally and importing it from anywhere

## Install

```bash
cd benchmark_generator
pip install -e .        # or just add this dir to PYTHONPATH
```

## Quick use

```python
from benchmark_generator import get_task, BenchmarkDataset, Split

task = get_task("addition")
ds = BenchmarkDataset(task, seed=0)

# infinite, never-memorised training stream
train_stream = ds.train_stream()
example = next(train_stream)
print(example.prompt, "->", example.target)

# fixed, reproducible eval sets (same across every architecture ablation)
test_ood_set = ds.eval_set(Split.TEST_OOD, n=1000)
```

## CLI

```bash
python -m benchmark_generator.cli --task dyck --split test_ood --n 5
python -m benchmark_generator.cli --all --n 3
python -m benchmark_generator.cli --task kv --split train --n 5000 --out kv_train.jsonl
```

## Tests

```bash
pytest -q
```

Each task's generated target is independently recomputed from the prompt
in `tests/test_tasks.py` (arithmetic sum, sort check, bracket-matching
stack simulation, dict lookup, array indexing, nested function eval), plus
a check that `test_ood` difficulty is strictly harder than `train`.

## Adding a new task

Subclass `benchmark_generator.base.Task`, implement `sample_difficulty`
and `generate_one`, register it in `benchmark_generator/registry.py`.
