"""Function Composition (FUNC).

Paper spec: apply a chain of symbolic functions to an argument.
  - train/val/test: 2-hop compositions (depth 2-3)
  - test_ood: 3- and 4- hop -> generalised to depth 5-10
  - example: "f(x)=x+2; g(x)=2x; g(f(3)) =" -> "10"

Functions are drawn from {x+c, x-c, c*x} with small integer constants,
named f, g, h, ... in application order (f applied first / innermost,
matching the paper's example g(f(3))).
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from ..base import Split, Task

FUNC_NAMES = list(string.ascii_lowercase[5:16])  # f, g, h, i, ... (avoid x)


@dataclass
class FuncConfig:
    depth_choices: Dict[Split, Tuple[int, int]] = field(default_factory=lambda: {
        Split.TRAIN: (2, 3),
        Split.VAL: (2, 3),
        Split.TEST: (2, 3),
        Split.TEST_OOD: (5, 10),
    })
    const_range: Tuple[int, int] = (1, 9)
    start_range: Tuple[int, int] = (0, 9)
    ops: Tuple[str, ...] = ("+", "-", "*")


class FuncTask(Task):
    name = "func"

    def __init__(self, config: FuncConfig | None = None):
        self.config = config or FuncConfig()
        assert len(FUNC_NAMES) >= max(self.config.depth_choices[s][1] for s in Split), \
            "not enough function-name letters for the requested max depth"

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        depth = rng.randint(*self.config.depth_choices[split])
        return {"depth": depth}

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        depth = params["depth"]
        names = FUNC_NAMES[:depth]

        definitions = []
        fns = {}
        for name in names:
            op = rng.choice(self.config.ops)
            c = rng.randint(*self.config.const_range)
            if op == "+":
                expr, fn = f"x+{c}", (lambda v, c=c: v + c)
            elif op == "-":
                expr, fn = f"x-{c}", (lambda v, c=c: v - c)
            else:
                expr, fn = f"{c}x", (lambda v, c=c: v * c)
            definitions.append(f"{name}(x)={expr}")
            fns[name] = fn

        x0 = rng.randint(*self.config.start_range)
        val = x0
        call_expr = str(x0)
        for name in names:  # innermost first, matches example g(f(3))
            val = fns[name](val)
            call_expr = f"{name}({call_expr})"

        prompt = "; ".join(definitions) + f"; {call_expr} ="
        target = str(val)
        return prompt, target
