import pytest

LABELS = ["card_arrival", "lost_or_stolen_card", "exchange_rate", "top_up_failed"]


@pytest.fixture
def labels():
    return list(LABELS)


@pytest.fixture
def rows():
    out = []
    for i in range(40):
        lab = LABELS[i % 4]
        out.append({"text": f"question {i} about {lab.replace('_', ' ')}", "label": lab})
    return out


def make_result(name, mode, f1, acc=None, inv=0.0, ts="2026-01-01T00:00:00+00:00", **extra):
    return {
        "run_name": name,
        "mode": mode,
        "strategy": None if mode == "finetuned" else name,
        "model": "m",
        "eval_size": 10,
        "seed": 1,
        "timestamp": ts,
        "metrics": {
            "accuracy": acc if acc is not None else f1,
            "accuracy_ci_low": 0.0,
            "accuracy_ci_high": 1.0,
            "macro_f1": f1,
            "invalid_rate": inv,
            "avg_latency_ms": 10.0,
            "avg_prompt_tokens": 100.0,
        },
        "top_confusions": [{"true": "a", "pred": "b", "count": 2}],
        **extra,
    }
