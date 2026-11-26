"""Dataset loading and deterministic splitting for Banking77."""

from __future__ import annotations

import random
from dataclasses import dataclass

Row = dict  # {"text": str, "label": str}


@dataclass
class DataBundle:
    labels: list[str]
    pool: list[Row]  # all non-validation train rows (source of few-shot examples)
    train: list[Row]  # subsample of pool used for fine-tuning
    val: list[Row]  # held out from train, used for eval_loss only
    eval: list[Row]  # test split subsample, used for ALL reported metrics


def subsample(rows: list[Row], n: int | None, seed: int) -> list[Row]:
    if n is None or n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(rows)), n))
    return [rows[i] for i in idx]


def split_train_val(rows: list[Row], val_size: int, seed: int) -> tuple[list[Row], list[Row]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[val_size:], shuffled[:val_size]


def build_bundle(train_all: list[Row], test: list[Row], labels: list[str], cfg: dict) -> DataBundle:
    d, seed = cfg["data"], cfg["seed"]
    pool, val = split_train_val(train_all, d.get("val_size", 200), seed)
    return DataBundle(
        labels=labels,
        pool=pool,
        train=subsample(pool, d.get("train_size"), seed),
        val=val,
        eval=subsample(test, d.get("eval_size"), seed),
    )


def load_banking77(cfg: dict) -> DataBundle:
    from datasets import load_dataset  # lazy: keeps CPU-only CI light

    ds = load_dataset(cfg["data"]["dataset"])
    labels = list(ds["train"].features["label"].names)

    def to_rows(split) -> list[Row]:
        return [{"text": r["text"], "label": labels[r["label"]]} for r in split]

    return build_bundle(to_rows(ds["train"]), to_rows(ds["test"]), labels, cfg)
