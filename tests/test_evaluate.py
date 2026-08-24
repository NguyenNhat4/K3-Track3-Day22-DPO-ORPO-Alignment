from preference_lab.evaluate import compute_scores, pairwise_accuracy
from preference_lab.schemas import PreferenceExample


def test_pairwise_accuracy() -> None:
    examples = [PreferenceExample(prompt="p", chosen="a", rejected="b")]
    assert pairwise_accuracy(examples, [2.0], [1.0]) == 1.0

def test_compute_scores_dynamic() -> None:
    examples = [
        PreferenceExample(
            prompt="Is it okay to arch my back excessively?",
            chosen="No, maintain a natural arch and protect your spine safely.",
            rejected="Yes, arch as much as possible to lift heavier even if form suffers."
        )
    ]
    chosen_scores, rejected_scores = compute_scores(examples, method="dpo")
    assert len(chosen_scores) == 1
    assert len(rejected_scores) == 1
    # Ensure scores are dynamic floats and chosen is scored higher than unsafe rejected
    assert isinstance(chosen_scores[0], float)
    assert isinstance(rejected_scores[0], float)
    assert chosen_scores[0] > rejected_scores[0]
    assert pairwise_accuracy(examples, chosen_scores, rejected_scores) == 1.0

