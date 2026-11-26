"""Output normalisation and metrics. Invalid generations count as wrong, never silently dropped."""

from __future__ import annotations

import difflib
import re

import numpy as np
from sklearn.metrics import f1_score

INVALID = "__invalid__"
_PREFIX = re.compile(r"^(label|intent|answer)\s*[:=]\s*", re.IGNORECASE)


def normalize_label(raw: str | None, labels: list[str]) -> str | None:
    """Map a raw generation to a known label, or None if it can't be matched confidently."""
    if not raw or not raw.strip():
        return None
    text = raw.strip().splitlines()[0]
    text = _PREFIX.sub("", text.strip())
    text = text.strip().strip("`'\".,:;!()[]{}* ").lower()
    text = re.sub(r"[\s\-]+", "_", text)
    lookup = {lab.lower(): lab for lab in labels}
    if text in lookup:
        return lookup[text]
    close = difflib.get_close_matches(text, list(lookup), n=1, cutoff=0.85)
    return lookup[close[0]] if close else None


def bootstrap_ci(correct: list[bool], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(correct, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def compute_metrics(
    y_true: list[str], raw_preds: list[str], labels: list[str]
) -> tuple[dict, list[str]]:
    preds = [normalize_label(r, labels) or INVALID for r in raw_preds]
    correct = [t == p for t, p in zip(y_true, preds, strict=True)]
    present = sorted(set(y_true))  # only score classes that appear in the eval set
    lo, hi = bootstrap_ci(correct)
    metrics = {
        "accuracy": float(np.mean(correct)),
        "accuracy_ci_low": lo,
        "accuracy_ci_high": hi,
        "macro_f1": float(
            f1_score(y_true, preds, labels=present, average="macro", zero_division=0)
        ),
        "invalid_rate": float(np.mean([p == INVALID for p in preds])),
    }
    return metrics, preds
