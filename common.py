"""Pure-Python helpers (no torch) so they can be unit-tested in CI."""

import difflib

TARGET_PREFIX = " "


def short_prompt(text: str) -> str:
    """Prompt used for the FINE-TUNED model (the task is baked into the weights)."""
    return f"Query: {text}\nIntent:"


def long_prompt(text: str, labels: list[str]) -> str:
    """Prompt-engineered BASELINE: the model must be told the task and all 77 labels every time."""
    return (
        "You are a banking intent classifier. Choose exactly ONE label from this list and output only the label.\n"
        + "\n".join(labels)
        + f"\n\nQuery: {text}\nIntent:"
    )


def parse_prediction(output: str, labels: list[str]) -> str | None:
    first = output.strip().split("\n")[0].strip().strip(".").lower()
    if first in labels:
        return first
    match = difflib.get_close_matches(first, labels, n=1, cutoff=0.7)
    return match[0] if match else None


def accuracy(preds: list[str | None], golds: list[str]) -> float:
    return sum(p == g for p, g in zip(preds, golds, strict=False)) / max(len(golds), 1)
