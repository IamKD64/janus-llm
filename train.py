"""LoRA / QLoRA fine-tune of a small open LLM on Banking77 intent classification.
Colab/Kaggle T4 (free):  python train.py --qlora
CPU smoke test:          python train.py --max-train 64 --epochs 1 --model Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from common import TARGET_PREFIX, short_prompt
from data import load_banking77


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="outputs/banking77-lora")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-train", type=int, default=0, help="0 = use all 10k examples")
    ap.add_argument("--qlora", action="store_true", help="4-bit base model (needs CUDA GPU)")
    args = ap.parse_args()

    train, _, labels = load_banking77()
    if args.max_train:
        train = train[: args.max_train]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token = tok.pad_token or tok.eos_token

    def encode(ex):
        p = tok(short_prompt(ex["text"]), add_special_tokens=False)["input_ids"]
        t = tok(TARGET_PREFIX + ex["label"], add_special_tokens=False)["input_ids"] + [
            tok.eos_token_id
        ]
        return {"input_ids": p + t, "labels": [-100] * len(p) + t}  # loss only on the label tokens

    enc = [encode(e) for e in train]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {
            "input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
            "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
            "attention_mask": torch.tensor(
                [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
            ),
        }

    cuda = torch.cuda.is_available()
    kwargs = {"torch_dtype": torch.float16 if cuda else torch.float32}
    if args.qlora:
        assert cuda, "--qlora needs a CUDA GPU (use free Colab/Kaggle T4)"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_strategy="no",
        fp16=cuda,
        report_to="none",
        remove_unused_columns=False,
        optim="paged_adamw_8bit" if args.qlora else "adamw_torch",
    )

    mlflow.set_experiment("banking77-lora")
    with mlflow.start_run(run_name=Path(args.out).name):
        mlflow.log_params({**vars(args), "n_train": len(enc), "n_labels": len(labels)})
        trainer = Trainer(model=model, args=targs, train_dataset=enc, data_collator=collate)
        result = trainer.train()
        mlflow.log_metric("train_loss", result.training_loss)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        Path(args.out, "labels.json").write_text(json.dumps(labels))
        mlflow.log_artifacts(args.out, artifact_path="adapter")
    print("Saved adapter to", args.out)


if __name__ == "__main__":
    main()
