from __future__ import annotations

from pathlib import Path

import typer
from rich import print

from .config import load_config
from .data import load_jsonl
from .evaluate import write_metrics
from .trainers import PreferenceTrainer, TrainingConfig

app = typer.Typer(help="Preference alignment lab CLI")

@app.command()
def validate(data: Path = typer.Argument(Path("data/sample_preferences.jsonl"), help="Path to JSONL preference data")) -> None:
    """Validate preference dataset schema and syntax."""
    examples = load_jsonl(data)
    print(f"[green]Successfully loaded {len(examples)} preference examples from {data}[/green]")

@app.command()
def evaluate(
    config: Path = typer.Option(Path("configs/local.yaml"), "--config", "-c", help="Path to YAML config file")
) -> None:
    """Evaluate pairwise accuracy and save metrics JSON."""
    cfg = load_config(config)
    examples = load_jsonl(cfg["paths"]["train_data"])
    training_kwargs = cfg.get("training", {})
    if "output_dir" not in training_kwargs and "paths" in cfg and "output_dir" in cfg["paths"]:
        training_kwargs["output_dir"] = cfg["paths"]["output_dir"]
    trainer = PreferenceTrainer(TrainingConfig(**training_kwargs))
    metrics = trainer.evaluate(examples)
    out = write_metrics(metrics, cfg["paths"]["output_dir"])
    print(f"[green]Wrote metrics to {out}[/green]")


if __name__ == "__main__":
    app()
