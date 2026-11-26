"""Model loading and batched greedy generation. torch/transformers are imported lazily."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenOutput:
    text: str
    prompt_tokens: int


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for batched generation with decoder-only models
    return tok


def compute_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_base_model(name: str, quantize: bool = True):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs: dict = {"device_map": "auto"}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
        )
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def attach_adapter(model, adapter_path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path)


def make_generate_fn(model, tokenizer, max_new_tokens: int):
    """Returns fn(list[messages]) -> list[GenOutput] (greedy decoding)."""
    import torch

    model.eval()

    def generate(batch_messages: list[list[dict]]) -> list[GenOutput]:
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        n_prompt = enc["attention_mask"].sum(dim=1).tolist()
        return [GenOutput(d.strip(), int(n)) for d, n in zip(decoded, n_prompt, strict=True)]

    return generate
