from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluate import compute_scores, pairwise_accuracy
from .losses import dpo_loss, orpo_loss
from .schemas import PreferenceExample


@dataclass(frozen=True)
class TrainingConfig:
    method: str = "dpo"
    beta: float = 0.1
    lambda_orpo: float = 0.1
    max_length: int = 512
    batch_size: int = 2
    output_dir: str = "outputs"

class PreferenceTrainer:
    """Interface and trainer for DPO/ORPO training and evaluation implementations."""
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def train(self) -> dict[str, float]:
        """Simulate training step and return output loss metrics."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.config.method.lower() == "dpo":
            loss = dpo_loss(
                policy_chosen_logps=np.array([-0.5, -0.4]),
                policy_rejected_logps=np.array([-1.5, -1.2]),
                ref_chosen_logps=np.array([-0.6, -0.5]),
                ref_rejected_logps=np.array([-1.0, -0.9]),
                beta=self.config.beta,
            )
        elif self.config.method.lower() == "orpo":
            loss = orpo_loss(
                sft_nll=np.array([0.5, 0.4]),
                chosen_logps=np.array([-0.5, -0.4]),
                rejected_logps=np.array([-1.5, -1.2]),
                lambda_orpo=self.config.lambda_orpo,
            )
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        return {"train_loss": float(loss)}

    def predict_scores(
        self, examples: list[PreferenceExample]
    ) -> tuple[list[float], list[float]]:
        """Compute model preference scores for chosen and rejected responses."""
        return compute_scores(
            examples,
            method=self.config.method,
            beta=self.config.beta,
            lambda_orpo=self.config.lambda_orpo,
        )

    def evaluate(self, examples: list[PreferenceExample]) -> dict[str, float]:
        """Evaluate preference model accuracy and loss metrics."""
        chosen_scores, rejected_scores = self.predict_scores(examples)
        acc = pairwise_accuracy(examples, chosen_scores, rejected_scores)
        metrics = self.train()
        metrics["pairwise_accuracy"] = acc
        return metrics

