"""Prompt strategies. The prompt baselines are deliberately strong so the comparison is fair.

strategies
  zero_shot      label list + instruction
  few_shot       label list + k worked examples as chat turns
  one_per_label  label list + one worked example for EVERY label (long, strongest baseline)
  finetuned      short instruction only; label knowledge lives in the adapter weights
"""

from __future__ import annotations

import random

from .data import Row

STRATEGIES = ("zero_shot", "few_shot", "one_per_label", "finetuned")

LABEL_SYSTEM = (
    "You are an intent classifier for a retail bank's customer support. "
    "Classify the customer's message into exactly one of the intent labels below. "
    "Answer with the label only, no explanation.\n\nLabels:\n{labels}"
)
FT_SYSTEM = (
    "You are an intent classifier for a retail bank's customer support. "
    "Classify the customer's message into exactly one of the following "
    "intent labels. Return only the exact label, with no explanation.\n\n"
    "Labels:\n{labels}"
)


def select_shots(pool: list[Row], k: int, seed: int) -> list[Row]:
    """k examples with distinct labels, deterministic."""
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    shots, seen = [], set()
    for r in shuffled:
        if r["label"] not in seen:
            shots.append(r)
            seen.add(r["label"])
        if len(shots) == k:
            break
    return shots


def one_per_label(pool: list[Row], labels: list[str], seed: int) -> list[Row]:
    rng = random.Random(seed)
    shuffled = list(pool)
    rng.shuffle(shuffled)
    first: dict[str, Row] = {}
    for r in shuffled:
        first.setdefault(r["label"], r)
    return [first[lab] for lab in labels if lab in first]


class PromptBuilder:
    def __init__(
        self,
        strategy: str,
        labels: list[str],
        pool: list[Row] | None = None,
        k: int = 5,
        seed: int = 42,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
        if strategy in ("few_shot", "one_per_label") and not pool:
            raise ValueError(f"{strategy} needs a pool of training examples")
        self.strategy, self.labels = strategy, labels
        label_block = "\n".join(f"- {lab}" for lab in labels)
        self.shots: list[Row] = []
        if strategy == "few_shot":
            self.shots = select_shots(pool, k, seed)
        system = LABEL_SYSTEM.format(labels=label_block)
        if strategy == "one_per_label":
            examples = one_per_label(pool, labels, seed)
            system += "\n\nExamples:\n" + "\n".join(
                f'"{r["text"]}" -> {r["label"]}' for r in examples
            )
        self.system = (
            FT_SYSTEM.format(labels=label_block)
            if strategy == "finetuned"
            else system
        )

    def messages(self, text: str) -> list[dict]:
        msgs = [{"role": "system", "content": self.system}]
        for shot in self.shots:
            msgs.append({"role": "user", "content": shot["text"]})
            msgs.append({"role": "assistant", "content": shot["label"]})
        msgs.append({"role": "user", "content": text})
        return msgs
