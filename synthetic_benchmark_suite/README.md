# Synthetic Benchmark Generator

A synthetic dataset (benchmark) generator for testing the algorithmic generalization of language models (based on the Transformer length generalization paper). 

The project generates data for 6 different algorithmic tasks. Each task supports a base difficulty (`train`, `val`, `test`) and an increased difficulty regime to test Out-Of-Distribution generalization (`test_ood`), where sequence length or nesting depth is scaled up.

## Project Structure

```text
benchmark_generator/
 ├── base.py            # Core architecture and CLI logic
 ├── addition.py        # Task: Long Addition
 ├── composition.py     # Task: Function Composition
 ├── dyck.py            # Task: Dyck Completion
 ├── indexing.py        # Task: Array Indexing
 ├── keyvalue.py        # Task: Key-Value Retrieval
 └── sorting.py         # Task: Array Sorting
```

## File Descriptions

*   **`base.py`** — The core architecture. It contains the base abstract `Task` class, dataclasses for storing generated examples, and a shared `run_task_cli` function that provides a unified command-line interface for all generators. It does not generate data on its own but is utilized by all other scripts.
*   **`addition.py`** (Long Addition) — Generates examples of digit-by-digit addition of two large numbers. The numbers are presented in a reversed-digit format for compatibility with RASP-L algorithmic setups.
*   **`composition.py`** (Function Composition) — Generates chained sequences of mathematical functions (e.g., $f(x)=x+2$, $g(x)=3x$) and computes the final value of a nested call (e.g., $g(f(2))$).
*   **`dyck.py`** (Dyck Completion) — Generates prefixes of random balanced bracket sequences using 3 types of brackets. The task is to predict the minimal required closing suffix in LIFO order.
*   **`indexing.py`** (Indexing) — The task is to extract an element from a randomly generated array at a specified integer index.
*   **`keyvalue.py`** (Key-Value Retrieval) — Generates a dictionary of key-value pairs (with fixed zero-padding widths to avoid tokenization artifacts) and queries the value for a randomly selected key.
*   **`sorting.py`** (Array Sorting) — Generates sequences of random integers (from 0 to 99) and their corresponding arrays sorted in ascending order.

## How to Run the Code

Each generator is a fully independent executable script. They all support the exact same command-line arguments.

### Basic Run
Prints 10 examples from the `train` split directly to your terminal:
```bash
python addition.py
```

### Generating Complex Examples (OOD)
Use the `--split` flag to change the difficulty regime (available: `train`, `val`, `test`, `test_ood`). 
To change the number of generated examples, use the `--n` flag:
```bash
python keyvalue.py --split test_ood --n 5
```

### Saving the Dataset to a File
To generate a complete dataset and save it in `.jsonl` format, use the `--out` flag:
```bash
python dyck.py --split train --n 10000 --out dyck_train.jsonl
```

### Available CLI Arguments:
*   `--split` *(str)*: Select the difficulty split (`train`, `val`, `test`, `test_ood`). Default is `train`.
*   `--n` *(int)*: The number of examples to generate. Default is 10.
*   `--seed` *(int)*: Random seed for reproducibility. Default is 0.
*   `--out` *(str)*: File path to save the output (in JSON Lines format). If not provided, the output is printed to the console.