from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from .schemas import PreferenceExample


def load_jsonl(path: str | Path) -> list[PreferenceExample]:
    """Load preference examples from JSONL file with line-numbered error tracking."""
    examples: list[PreferenceExample] = []
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with filepath.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_idx}: Invalid JSON syntax - {e}") from e

            try:
                example = PreferenceExample.model_validate(data)
            except ValidationError as e:
                raise ValueError(f"Line {line_idx}: Invalid schema - {e}") from e
            examples.append(example)
    return examples

def split_by_prompt(
    examples: list[PreferenceExample], validation_ratio: float = 0.2, seed: int = 42
) -> tuple[list[PreferenceExample], list[PreferenceExample]]:
    """Split examples by prompt to avoid data leakage using deterministic shuffling."""
    if not examples:
        return [], []

    prompt_to_examples = defaultdict(list)
    for ex in examples:
        prompt_to_examples[ex.prompt].append(ex)

    unique_prompts = sorted(prompt_to_examples.keys())
    random.Random(seed).shuffle(unique_prompts)

    val_cut = int(len(unique_prompts) * validation_ratio)
    val_prompts = set(unique_prompts[:val_cut])

    train, val = [], []
    for prompt in unique_prompts:
        if prompt in val_prompts:
            val.extend(prompt_to_examples[prompt])
        else:
            train.extend(prompt_to_examples[prompt])

    return train, val
