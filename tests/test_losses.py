import numpy as np

from preference_lab.losses import dpo_loss, orpo_loss


def test_dpo_loss_computation() -> None:
    loss = dpo_loss(
        policy_chosen_logps=np.array([-0.5]),
        policy_rejected_logps=np.array([-1.5]),
        ref_chosen_logps=np.array([-0.6]),
        ref_rejected_logps=np.array([-1.0]),
        beta=0.1,
    )
    assert isinstance(loss, float)
    assert loss > 0.0

def test_orpo_loss_computation() -> None:
    loss = orpo_loss(
        sft_nll=np.array([1.0]),
        chosen_logps=np.array([-0.5]),
        rejected_logps=np.array([-1.5]),
        lambda_orpo=0.1,
    )
    assert isinstance(loss, float)
    assert loss > 0.0
