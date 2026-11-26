"""Evaluate base model with a long engineered prompt vs the fine-tuned adapter with a short prompt.
  python evaluate.py --mode baseline                                  # prompt engineering only
  python evaluate.py --mode finetuned --adapter outputs/banking77-lora
Reports accuracy, latency/query and prompt length (a proxy for per-request cost)."""
import argparse
import json
import time
from pathlib import Path

import mlflow
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import accuracy, long_prompt, parse_prediction, short_prompt
from data import load_banking77


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--mode", choices=["baseline", "finetuned"], required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()

    _, test, labels = load_banking77()
    test = test[: args.n]
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if cuda else torch.float32, device_map={"": 0} if cuda else None)
    if args.mode == "finetuned":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    prompts = [long_prompt(e["text"], labels) if args.mode == "baseline" else short_prompt(e["text"]) for e in test]
    preds, tokens, t0 = [], 0, time.perf_counter()
    for i in range(0, len(prompts), args.bs):
        batch = tok(prompts[i : i + args.bs], return_tensors="pt", padding=True).to(model.device)
        tokens += int(batch["attention_mask"].sum())
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=12, do_sample=False, pad_token_id=tok.pad_token_id)
        preds += [parse_prediction(t, labels) for t in tok.batch_decode(out[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)]
    elapsed = time.perf_counter() - t0

    res = {"mode": args.mode, "model": args.model, "n": len(test),
           "accuracy": round(accuracy(preds, [e["label"] for e in test]), 4),
           "avg_prompt_tokens": round(tokens / len(test), 1), "sec_per_query": round(elapsed / len(test), 3)}
    print(json.dumps(res, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{args.mode}.json").write_text(json.dumps(res, indent=2))
    mlflow.set_experiment("banking77-lora")
    with mlflow.start_run(run_name=f"eval-{args.mode}"):
        mlflow.log_params({"mode": args.mode, "model": args.model, "n": len(test)})
        mlflow.log_metrics({k: v for k, v in res.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
