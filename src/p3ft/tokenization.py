"""Turn (messages, target label) into causal-LM training examples with prompt tokens masked."""

from __future__ import annotations

from .data import Row
from .prompts import PromptBuilder


def encode_example(tokenizer, messages: list[dict], target: str, max_len: int) -> dict | None:
    """Loss is computed ONLY on the label tokens (+ EOS). Returns None if too long."""
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    ids = p_ids + t_ids
    if len(ids) > max_len:
        return None
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(p_ids) + t_ids,
    }


def build_dataset(
    tokenizer, rows: list[Row], builder: PromptBuilder, max_len: int
) -> tuple[list[dict], int]:
    examples, dropped = [], 0
    for r in rows:
        ex = encode_example(tokenizer, builder.messages(r["text"]), r["label"], max_len)
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    return examples, dropped
