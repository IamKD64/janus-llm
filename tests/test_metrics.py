import pytest

from p3ft.metrics import INVALID, bootstrap_ci, compute_metrics, normalize_label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("card_arrival", "card_arrival"),
        ("  Card_Arrival.  ", "card_arrival"),
        ("`lost_or_stolen_card`", "lost_or_stolen_card"),
        ("Label: exchange_rate", "exchange_rate"),
        ("card arrival", "card_arrival"),
        ("top-up-failed\nbecause...", "top_up_failed"),
        ("card_arrivl", "card_arrival"),  # typo within fuzzy cutoff
        ("I think it is something else", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize(raw, expected, labels):
    assert normalize_label(raw, labels) == expected


def test_metrics_perfect_and_invalid(labels):
    y = ["card_arrival", "exchange_rate", "top_up_failed", "lost_or_stolen_card"]
    m, preds = compute_metrics(y, y, labels)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["invalid_rate"] == 0.0

    m, preds = compute_metrics(
        y, ["card_arrival", "garbage", "top_up_failed", "exchange_rate"], labels
    )
    assert m["accuracy"] == 0.5 and m["invalid_rate"] == 0.25
    assert INVALID in preds


def test_macro_f1_ignores_absent_classes(labels):
    y = ["card_arrival", "card_arrival"]
    m, _ = compute_metrics(y, y, labels)
    assert m["macro_f1"] == 1.0


def test_bootstrap_ci_bounds():
    lo, hi = bootstrap_ci([True] * 50 + [False] * 50)
    assert 0.35 < lo < 0.5 < hi < 0.65
    assert bootstrap_ci([]) == (0.0, 0.0)
