"""Evaluate prompt baselines or the fine-tuned adapter on the same held-out set.

python -m p3ft.evaluate --mode baseline                    # all configured strategies
python -m p3ft.evaluate --mode baseline --strategy few_shot
python -m p3ft.evaluate --mode finetuned --adapter outputs/qlora-banking77
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .metrics import compute_metrics
from .prompts import PromptBuilder
from .tracking import flatten, setup_mlflow


def run_eval(rows, labels, builder: PromptBuilder, generate_fn, batch_size: int) -> dict:
    """Pure evaluation loop (no torch): easy to unit-test with a mock generate_fn."""
    raws: list[str] = []
    prompt_tokens: list[int] = []
    gen_seconds = 0.0
    for i in range(0, len(rows), batch_size):
        batch = rows[i : i + batch_size]
        t0 = time.perf_counter()
        outs = generate_fn([builder.messages(r["text"]) for r in batch])
        gen_seconds += time.perf_counter() - t0
        raws += [o.text for o in outs]
        prompt_tokens += [o.prompt_tokens for o in outs]

    y_true = [r["label"] for r in rows]
    metrics, preds = compute_metrics(y_true, raws, labels)
    metrics["avg_latency_ms"] = 1000 * gen_seconds / max(len(rows), 1)  # amortised per sample
    metrics["avg_prompt_tokens"] = float(np.mean(prompt_tokens)) if prompt_tokens else 0.0
    confusions = Counter((t, p) for t, p in zip(y_true, preds, strict=True) if t != p)
    return {
        "metrics": metrics,
        "top_confusions": [
            {"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(10)
        ],
        "predictions": pd.DataFrame(
            {"text": [r["text"] for r in rows], "true": y_true, "raw": raws, "pred": preds}
        ),
    }


def save_and_log(
    mlflow, cfg: dict, run_name: str, mode: str, strategy: str | None, result: dict, extra: dict
) -> dict:
    results_dir = Path(cfg["results_dir"])
    (results_dir / "predictions").mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "predictions" / f"{run_name}.csv"
    result["predictions"].to_csv(csv_path, index=False)

    with mlflow.start_run(run_name=run_name, tags={"stage": "evaluation", "mode": mode}) as run:
        mlflow.log_params(
            {
                "mode": mode,
                "strategy": strategy or "finetuned",
                "model": cfg["model"]["name"],
                "eval_size": len(result["predictions"]),
                "seed": cfg["seed"],
                **flatten({"eval": cfg["eval"]}),
                **{k: v for k, v in extra.items() if v is not None},
            }
        )
        mlflow.log_metrics(result["metrics"])
        summary = {
            "run_name": run_name,
            "mode": mode,
            "strategy": strategy,
            "model": cfg["model"]["name"],
            "eval_size": len(result["predictions"]),
            "seed": cfg["seed"],
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": run.info.run_id,
            "metrics": result["metrics"],
            "top_confusions": result["top_confusions"],
            **extra,
        }
        json_path = results_dir / f"{run_name}.json"
        json_path.write_text(json.dumps(summary, indent=2))
        mlflow.log_artifact(str(csv_path), "predictions")
        mlflow.log_artifact(str(json_path))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--mode", choices=["baseline", "finetuned"], required=True)
    ap.add_argument("--strategy", default=None, help="baseline only; default = all in config")
    ap.add_argument("--adapter", default=None, help="adapter dir; default = train.output_dir")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    mlflow = setup_mlflow(cfg)

    from .data import load_banking77
    from .modeling import attach_adapter, load_base_model, load_tokenizer, make_generate_fn

    bundle = load_banking77(cfg)
    tok = load_tokenizer(cfg["model"]["name"])
    ecfg = cfg["eval"]

    if args.mode == "baseline":
        strategies = [args.strategy] if args.strategy else ecfg["strategies"]
        model = load_base_model(cfg["model"]["name"], quantize=ecfg.get("quantize_baseline", True))
        gen = make_generate_fn(model, tok, ecfg["max_new_tokens"])
        for s in strategies:
            builder = PromptBuilder(
                s, bundle.labels, bundle.pool, ecfg.get("few_shot_k", 5), cfg["seed"]
            )
            print(f"\n=== baseline: {s} ===")
            result = run_eval(bundle.eval, bundle.labels, builder, gen, ecfg["batch_size"])
            summary = save_and_log(mlflow, cfg, f"baseline-{s}", "baseline", s, result, {})
            print(json.dumps(summary["metrics"], indent=2))
    else:
        adapter = Path(args.adapter or cfg["train"]["output_dir"])
        meta_path = adapter / "train_meta.json"
        train_run_id = json.loads(meta_path.read_text())["run_id"] if meta_path.exists() else None
        model = load_base_model(cfg["model"]["name"], quantize=cfg["qlora"]["quantize"])
        model = attach_adapter(model, str(adapter))
        gen = make_generate_fn(model, tok, ecfg["max_new_tokens"])
        builder = PromptBuilder("finetuned", bundle.labels)
        print("\n=== fine-tuned QLoRA adapter ===")
        result = run_eval(bundle.eval, bundle.labels, builder, gen, ecfg["batch_size"])
        summary = save_and_log(
            mlflow,
            cfg,
            "finetuned-qlora",
            "finetuned",
            None,
            result,
            {"adapter_path": str(adapter), "train_run_id": train_run_id},
        )
        print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
