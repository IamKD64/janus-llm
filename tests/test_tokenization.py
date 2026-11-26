from p3ft.prompts import PromptBuilder
from p3ft.tokenization import build_dataset, encode_example


class FakeTok:
    eos_token = "<E>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"{m['role']}:{m['content']}|" for m in messages)
        return s + ("assistant:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_masked_and_target_supervised():
    tok = FakeTok()
    msgs = [{"role": "user", "content": "hi"}]
    ex = encode_example(tok, msgs, "card_arrival", max_len=200)
    n_prompt = len(tok.apply_chat_template(msgs, add_generation_prompt=True))
    assert ex["labels"][:n_prompt] == [-100] * n_prompt
    supervised = ex["labels"][n_prompt:]
    assert "".join(map(chr, supervised)) == "card_arrival<E>"
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_too_long_dropped(rows, labels):
    builder = PromptBuilder("finetuned", labels)
    ds, dropped = build_dataset(FakeTok(), rows, builder, max_len=10)
    assert ds == [] and dropped == len(rows)
    ds, dropped = build_dataset(FakeTok(), rows, builder, max_len=500)
    assert len(ds) == len(rows) and dropped == 0
