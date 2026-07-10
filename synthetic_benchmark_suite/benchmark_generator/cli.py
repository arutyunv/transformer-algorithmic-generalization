"""Quick CLI to preview or dump generated examples.

Examples
--------
    python -m benchmark_generator.cli --task addition --split test_ood --n 5
    python -m benchmark_generator.cli --task kv --split train --n 1000 --out kv_train.jsonl
    python -m benchmark_generator.cli --all --n 3          # preview every task
"""
from __future__ import annotations

import argparse
import json
import sys

from .base import Split
from .dataset import BenchmarkDataset
from .registry import TASK_REGISTRY, all_tasks


def _dump(examples, out_path: str | None):
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
        print(f"wrote {len(examples)} examples -> {out_path}", file=sys.stderr)
    else:
        for ex in examples:
            print(f"[{ex.task}/{ex.split}] {ex.prompt!r} -> {ex.target!r}  meta={ex.meta}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=sorted(TASK_REGISTRY), help="task name")
    p.add_argument("--all", action="store_true", help="preview every registered task")
    p.add_argument("--split", choices=[s.value for s in Split], default="train")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="write JSONL here instead of printing")
    args = p.parse_args()

    if not args.task and not args.all:
        p.error("pass --task <name> or --all")

    split = Split(args.split)

    if args.all:
        for name, task in all_tasks().items():
            ds = BenchmarkDataset(task, seed=args.seed)
            if split == Split.TRAIN:
                stream = ds.train_stream()
                examples = [next(stream) for _ in range(args.n)]
            else:
                examples = ds.eval_set(split, args.n)
            _dump(examples, None)
        return

    ds = BenchmarkDataset(TASK_REGISTRY[args.task](), seed=args.seed)
    if split == Split.TRAIN:
        stream = ds.train_stream()
        examples = [next(stream) for _ in range(args.n)]
    else:
        examples = ds.eval_set(split, args.n)
    _dump(examples, args.out)


if __name__ == "__main__":
    main()
