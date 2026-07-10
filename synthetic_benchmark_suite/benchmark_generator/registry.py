"""Name -> Task class registry, so training code can pick a task by string
(e.g. from a config file) without importing every task module by hand."""
from __future__ import annotations

from typing import Dict, Type

from .base import Task
from .tasks import (
    AdditionTask,
    SortingTask,
    DyckTask,
    KVTask,
    IndexTask,
    FuncTask,
)

TASK_REGISTRY: Dict[str, Type[Task]] = {
    "addition": AdditionTask,
    "sorting": SortingTask,
    "dyck": DyckTask,
    "kv": KVTask,
    "indexing": IndexTask,
    "func": FuncTask,
}


def get_task(name: str) -> Task:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()


def all_tasks() -> Dict[str, Task]:
    return {name: cls() for name, cls in TASK_REGISTRY.items()}
