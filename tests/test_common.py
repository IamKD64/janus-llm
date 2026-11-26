import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import accuracy, long_prompt, parse_prediction, short_prompt

LABELS = ["card_arrival", "lost_or_stolen_card", "exchange_rate"]


def test_parse_exact_and_fuzzy_and_none():
    assert parse_prediction(" card_arrival\nextra", LABELS) == "card_arrival"
    assert parse_prediction("lost_or_stollen_card.", LABELS) == "lost_or_stolen_card"
    assert parse_prediction("banana", LABELS) is None


def test_prompts_and_accuracy():
    assert short_prompt("hi").endswith("Intent:")
    assert all(l in long_prompt("hi", LABELS) for l in LABELS)
    assert len(long_prompt("hi", LABELS)) > len(short_prompt("hi"))
    assert accuracy(["a", None, "c"], ["a", "b", "c"]) == 2 / 3


def test_bundled_banking77_and_cost():
    from cost_compare import estimate_cost
    from data import load_banking77

    train, test, labels = load_banking77()
    assert len(train) == 10003 and len(test) == 3080 and len(labels) == 77
    assert all(r["label"] in labels for r in test)
    # 1M requests x 100 input tokens at $1 per 1M tokens = $100
    assert estimate_cost(100, 0, 1.0, 5.0) == 100.0
