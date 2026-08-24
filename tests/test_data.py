import pytest
from pydantic import ValidationError

from preference_lab.data import load_jsonl, split_by_prompt
from preference_lab.schemas import PreferenceExample


def test_load_sample_data() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    assert len(examples) >= 2
    assert examples[0].chosen != examples[0].rejected

def test_split_returns_all_examples() -> None:
    examples = load_jsonl("data/sample_preferences.jsonl")
    train, val = split_by_prompt(examples, validation_ratio=0.5)
    assert len(train) + len(val) == len(examples)

def test_chosen_rejected_must_differ() -> None:
    with pytest.raises(ValidationError):
        PreferenceExample(prompt="p", chosen="  Hello World  ", rejected="hello world")

def test_load_jsonl_invalid_json_line(tmp_path: pytest.TempPathFactory) -> None:
    invalid_file = tmp_path / "invalid.jsonl"
    invalid_file.write_text('{"prompt": "valid", "chosen": "a", "rejected": "b"}\n{invalid_json}\n')
    with pytest.raises(ValueError, match="Line 2: Invalid JSON syntax"):
        load_jsonl(invalid_file)

def test_load_jsonl_invalid_schema_line(tmp_path: pytest.TempPathFactory) -> None:
    invalid_file = tmp_path / "invalid_schema.jsonl"
    invalid_file.write_text('{"prompt": "valid", "chosen": "a", "rejected": "b"}\n{"prompt": "p", "chosen": "same", "rejected": "SAME"}\n')
    with pytest.raises(ValueError, match="Line 2: Invalid schema"):
        load_jsonl(invalid_file)

def test_split_by_prompt_no_data_leakage() -> None:
    examples = [
        PreferenceExample(prompt="p1", chosen="c1", rejected="r1"),
        PreferenceExample(prompt="p1", chosen="c2", rejected="r2"),
        PreferenceExample(prompt="p2", chosen="c3", rejected="r3"),
        PreferenceExample(prompt="p3", chosen="c4", rejected="r4"),
        PreferenceExample(prompt="p4", chosen="c5", rejected="r5"),
    ]
    train, val = split_by_prompt(examples, validation_ratio=0.4, seed=42)
    train_prompts = {ex.prompt for ex in train}
    val_prompts = {ex.prompt for ex in val}
    assert train_prompts.isdisjoint(val_prompts)
    assert len(train) + len(val) == len(examples)

