"""Key-Value Retrieval (KV).

Paper spec: n key-value pairs followed by a query key -> retrieve the
matching value.
  - train/val/test: n = 16 and 64 pairs
  - test_ood: n = 256 and 1024 pairs
  - example: "k17:v93 k04:v12 QUERY k04 =" -> "v12"

Keys are sampled without replacement from a pool large enough to
guarantee uniqueness even at n=1024 (address-content separation only
makes sense if keys are unique).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from ..base import Split, Task


@dataclass
class KVConfig:
    n_choices: Dict[Split, Tuple[int, ...]] = field(default_factory=lambda: {
        Split.TRAIN: (16, 64),
        Split.VAL: (16, 64),
        Split.TEST: (16, 64),
        Split.TEST_OOD: (256, 1024),
    })
    value_range: Tuple[int, int] = (0, 999)
    # key pool multiplier relative to n, must stay unique at largest n (1024)
    key_pool_multiplier: int = 8


class KVTask(Task):
    name = "kv"

    def __init__(self, config: KVConfig | None = None):
        self.config = config or KVConfig()

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        n = rng.choice(self.config.n_choices[split])
        return {"n": n}

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        n = params["n"]
        pool_size = max(n * self.config.key_pool_multiplier, n + 1)
        keys = rng.sample(range(pool_size), n)
        lo, hi = self.config.value_range
        values = [rng.randint(lo, hi) for _ in range(n)]

        query_idx = rng.randrange(n)
        query_key, answer = keys[query_idx], values[query_idx]

        body = " ".join(f"k{k}:v{v}" for k, v in zip(keys, values))
        prompt = f"{body} QUERY k{query_key} ="
        target = f"v{answer}"
        return prompt, target
