import pytest

from p3ft.prompts import PromptBuilder, select_shots


def test_zero_shot(labels):
    msgs = PromptBuilder("zero_shot", labels).messages("hi")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert all(lab in msgs[0]["content"] for lab in labels)


def test_few_shot_distinct_labels(rows, labels):
    b = PromptBuilder("few_shot", labels, rows, k=3, seed=1)
    assert len({s["label"] for s in b.shots}) == 3
    msgs = b.messages("q")
    assert [m["role"] for m in msgs] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert msgs[-1]["content"] == "q"


def test_one_per_label_covers_all(rows, labels):
    system = PromptBuilder("one_per_label", labels, rows).system
    assert system.count(" -> ") == len(labels)


def test_finetuned_prompt_is_short_and_label_free(labels):
    system = PromptBuilder("finetuned", labels).system
    assert not any(lab in system for lab in labels)


def test_deterministic(rows):
    assert select_shots(rows, 4, 7) == select_shots(rows, 4, 7)


def test_errors(labels):
    with pytest.raises(ValueError):
        PromptBuilder("nope", labels)
    with pytest.raises(ValueError):
        PromptBuilder("few_shot", labels, pool=None)
