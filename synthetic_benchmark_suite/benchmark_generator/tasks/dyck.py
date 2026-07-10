"""Dyck-language Completion (DYCK).

Paper spec: k=3 bracket types. Given a prefix of a balanced-bracket
string, predict the *minimal* closing suffix (i.e. exactly the
brackets needed to close everything currently open, in LIFO order).

  - train/val/test: nesting depth 2-4, length <= 20
  - test_ood: nesting depth 8-16, length 21-40
  - example: "COMPLETE (()( =" -> "))"

We build the prompt with a random open/close walk bounded by a max
nesting depth and a target length, then read the target directly off
the open-bracket stack instead of continuing the random walk - this
guarantees the suffix really is the *minimal* one (only closes, no
extra opens), matching the paper's definition.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from ..base import Split, Task

PAIRS = {"(": ")", "[": "]", "{": "}"}
OPENS = list(PAIRS.keys())


@dataclass
class DyckConfig:
    depth_choices: Dict[Split, Tuple[int, int]] = field(default_factory=lambda: {
        Split.TRAIN: (2, 4),
        Split.VAL: (2, 4),
        Split.TEST: (2, 4),
        Split.TEST_OOD: (8, 16),
    })
    length_choices: Dict[Split, Tuple[int, int]] = field(default_factory=lambda: {
        Split.TRAIN: (10, 20),
        Split.VAL: (10, 20),
        Split.TEST: (10, 20),
        Split.TEST_OOD: (21, 40),
    })
    # probability of opening a new bracket rather than closing, when both
    # are legal moves. Higher -> deeper / more brackets left open.
    open_bias: float = 0.6


class DyckTask(Task):
    name = "dyck"

    def __init__(self, config: DyckConfig | None = None):
        self.config = config or DyckConfig()

    def sample_difficulty(self, split: Split, rng: random.Random) -> Dict[str, Any]:
        depth = rng.randint(*self.config.depth_choices[split])
        length = rng.randint(*self.config.length_choices[split])
        return {"depth": depth, "length": length}

    def generate_one(self, params: Dict[str, Any], rng: random.Random) -> Tuple[str, str]:
        depth_max = params["depth"]
        length = params["length"]

        stack: list[str] = []
        chars: list[str] = []
        for _ in range(length):
            can_open = len(stack) < depth_max
            can_close = len(stack) > 0
            if can_open and (not can_close or rng.random() < self.config.open_bias):
                c = rng.choice(OPENS)
                stack.append(c)
                chars.append(c)
            elif can_close:
                c = stack.pop()
                chars.append(PAIRS[c])
            else:
                break

        if not stack:  # force a non-trivial completion task
            c = rng.choice(OPENS)
            stack.append(c)
            chars.append(c)

        prompt = "COMPLETE " + "".join(chars) + " ="
        target = "".join(PAIRS[c] for c in reversed(stack))
        return prompt, target
