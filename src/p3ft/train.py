"""QLoRA fine-tuning (4-bit NF4 base + LoRA adapters), loss on label tokens only.

python -m p3ft.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .tracking import flatten, setup_mlflow


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    mlflow = setup_mlflow(cfg)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed

    from .data import load_banking77
    from .modeling import load_base_model, load_tokenizer
    from .prompts import PromptBuilder
    from .tokenization import build_dataset

    set_seed(cfg["seed"])
    t, q, mname = cfg["train"], cfg["qlora"], cfg["model"]["name"]
    out_dir = Path(t["output_dir"])

    bundle = load_banking77(cfg)
    tok = load_tokenizer(mname)
    builder = PromptBuilder("finetuned", bundle.labels)
    max_len = cfg["model"]["max_seq_len"]
    train_ds, d1 = build_dataset(tok, bundle.train, builder, max_len)
    val_ds, d2 = build_dataset(tok, bundle.val, builder, max_len)
    print(f"train={len(train_ds)} (dropped {d1})  val={len(val_ds)} (dropped {d2})")

    model = load_base_model(mname, quantize=q["quantize"])
    if q["quantize"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=q["r"],
            lora_alpha=q["alpha"],
            lora_dropout=q["dropout"],
            target_modules=q["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        output_dir=str(
            out_dir.parent / f"{out_dir.name}-checkpoints"
        ),  # keep out of the adapter dir
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"],
        warmup_ratio=t["warmup_ratio"],
        lr_scheduler_type=t["lr_scheduler"],
        logging_steps=t["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        optim="paged_adamw_8bit" if q["quantize"] else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to=["mlflow"],
        seed=cfg["seed"],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(
            tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        processing_class=tok,
    )

    with mlflow.start_run(run_name="qlora-train", tags={"stage": "training"}) as run:
        mlflow.log_params(
            flatten({k: cfg[k] for k in ("seed", "data", "model", "qlora", "train")}, "cfg.")
        )
        mlflow.log_params(
            {
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "trainable_params": trainable,
                "total_params": total,
            }
        )
        trainer.train()
        final_eval = trainer.evaluate()
        mlflow.log_metrics(
            {f"final_{k}": v for k, v in final_eval.items() if isinstance(v, (int, float))}
        )

        out_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        (out_dir / "labels.json").write_text(json.dumps(bundle.labels))
        (out_dir / "train_meta.json").write_text(
            json.dumps({"run_id": run.info.run_id, "base_model": mname})
        )
        mlflow.log_artifacts(str(out_dir), "adapter")
        print(f"Saved adapter to {out_dir}  (MLflow run {run.info.run_id})")


if __name__ == "__main__":
    main()
