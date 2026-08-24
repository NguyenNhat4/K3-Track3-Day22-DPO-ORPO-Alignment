from __future__ import annotations

import numpy as np


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log_sigmoid computation."""
    return np.where(x >= 0, -np.log1p(np.exp(-x)), x - np.log1p(np.exp(x)))

def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float = 0.1,
) -> float:
    """Compute batch DPO loss from sequence log probabilities."""
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps
    logits = policy_log_ratios - ref_log_ratios

    losses = -_log_sigmoid(beta * logits)
    return float(np.mean(losses))

def orpo_loss(
    sft_nll: np.ndarray,
    chosen_logps: np.ndarray,
    rejected_logps: np.ndarray,
    lambda_orpo: float = 0.1,
) -> float:
    """Compute ORPO objective = SFT NLL Loss + Odds Ratio Penalty."""
    chosen_log_odds = chosen_logps - np.log1p(-np.exp(chosen_logps) + 1e-7)
    rejected_log_odds = rejected_logps - np.log1p(-np.exp(rejected_logps) + 1e-7)

    log_or = chosen_log_odds - rejected_log_odds
    or_penalty = -_log_sigmoid(log_or)

    total_loss = np.mean(sft_nll) + lambda_orpo * np.mean(or_penalty)
    return float(total_loss)
