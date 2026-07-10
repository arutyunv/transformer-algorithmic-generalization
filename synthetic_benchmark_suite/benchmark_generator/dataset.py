"""Wraps a Task with the train/eval data-regime split used across all
ablations in the paper:

  - train:      infinite on-the-fly stream, never memorised
  - val/test:   fixed, reproducible set (same difficulty distribution as
                train) so every architecture in the ablation table is
                compared on the *same* held-out examples
  - test_ood:   fixed, reproducible set at the shifted difficulty axis
"""
from __future__ import annotations

from typing import Iterator, List

from .base import Example, Split, Task

# Distinct base seeds per split so val/test/test_ood never collide with
# each other or with the train stream, regardless of the user-provided seed.
_SPLIT_SEED_OFFSET = {
    Split.TRAIN: 0,
    Split.VAL: 1_000_003,
    Split.TEST: 2_000_003,
    Split.TEST_OOD: 3_000_003,
}


class BenchmarkDataset:
    def __init__(self, task: Task, seed: int = 0):
        self.task = task
        self.seed = seed

    def _seed_for(self, split: Split) -> int:
        return self.seed + _SPLIT_SEED_OFFSET[split]

    def train_stream(self) -> Iterator[Example]:
        """Unbounded generator of fresh training examples."""
        return self.task.infinite_stream(Split.TRAIN, seed=self._seed_for(Split.TRAIN))

    def eval_set(self, split: Split, n: int) -> List[Example]:
        """Fixed, reproducible set for val / test / test_ood."""
        if split == Split.TRAIN:
            raise ValueError("Use train_stream() for the train split.")
        return self.task.generate_many(split, n=n, seed=self._seed_for(split))
