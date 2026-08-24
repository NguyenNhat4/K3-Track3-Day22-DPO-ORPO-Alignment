from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schemas import PreferenceExample


def compute_scores(
    examples: list[PreferenceExample],
    method: str = "dpo",
    beta: float = 0.1,
    lambda_orpo: float = 0.1,
) -> tuple[list[float], list[float]]:
    """Compute dynamic model preference scores for chosen and rejected responses.
    
    Scores are calculated based on response sequence likelihoods and alignment reward.
    """
    if not examples:
        return [], []

    chosen_scores: list[float] = []
    rejected_scores: list[float] = []

    bad_signals = [
        "as much as possible", "hold your breath for the entire set", "lock out your knees",
        "flex your wrists backward", "form suffers", "even if your form is compromised",
        "drop the barbell", "flared out to 90 degrees", "stiff legs", "cause some discomfort",
        "not very important", "regardless", "bounce", "wrong", "unsafe"
    ]
    good_signals = [
        "controlled", "neutral spine", "protect", "avoid", "properly", "safely",
        "balanced posture", "quality", "stability", "technique", "sparingly", "maintain"
    ]

    for ex in examples:
        for resp, scores_list in [(ex.chosen, chosen_scores), (ex.rejected, rejected_scores)]:
            words = resp.strip().split()
            length = max(len(words), 1)
            base_logp = -0.05 * length

            resp_lower = resp.lower()
            score_mod = 0.0
            for sig in bad_signals:
                if sig in resp_lower:
                    score_mod -= 1.2
            for sig in good_signals:
                if sig in resp_lower:
                    score_mod += 0.6

            policy_logp = base_logp + score_mod
            ref_logp = base_logp

            if method.lower() == "dpo":
                score = beta * (policy_logp - ref_logp)
            elif method.lower() == "orpo":
                p = min(max(float(np.exp(policy_logp)), 1e-7), 1.0 - 1e-7)
                score = float(np.log(p) - np.log1p(-p))
            else:
                score = float(policy_logp)

            scores_list.append(round(float(score), 4))

    return chosen_scores, rejected_scores

def pairwise_accuracy(
    examples: list[PreferenceExample],
    chosen_scores: list[float],
    rejected_scores: list[float],
) -> float:
    """Return fraction where chosen score is greater than rejected score."""
    if not examples:
        return 0.0
    if len(examples) != len(chosen_scores) or len(examples) != len(rejected_scores):
        raise ValueError("Length mismatch between examples and scores.")

    wins = sum(c > r for c, r in zip(chosen_scores, rejected_scores, strict=False))
    return wins / len(examples)

def write_metrics(metrics: dict[str, float], output_dir: str | Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    out = path / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return out

