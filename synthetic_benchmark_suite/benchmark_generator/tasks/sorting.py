"""Array Sorting (SORT).

Paper spec: integers drawn uniformly from [0, 99].
  - train/val/test: length 8 and 16
  - test_ood: length 32 and 64
  - example: "7 2 9 1 4 =" -> "1 2 4 7 9"
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from ..base import Split, Task


@dataclass
class SortingConfig:
    length_choices: Dict[Split, Tuple[int, ...]] = field(default_factory=lambda: {
        Split.TRAIN: (8, 16),
        Split.VAL: (8, 16),
        Split.TEST: (8, 16),
        Split.TEST_OOD: (32, 64),
    })
    value_range: Tuple[int, int] = (0, 99)


class SortingTask(Task):
    name = "sorting"

    def __init__(self, config: SortingConfig | None = None):
        self.config = config or SortingConfig()

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        length = rng.choice(self.config.length_choices[split])
        return {"length": length}

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        n = params["length"]
        lo, hi = self.config.value_range
        values = [rng.randint(lo, hi) for _ in range(n)]
        prompt = " ".join(map(str, values)) + " ="
        target = " ".join(map(str, sorted(values)))
        return prompt, target
