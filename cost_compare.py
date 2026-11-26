"""Token-cost view of prompt engineering vs fine-tuning, using REAL API prices from data/llm_models.csv.
Run after both evaluations:  python cost_compare.py
Compares the API cost of the long engineered prompt (baseline) with the short prompt the fine-tuned model needs,
at the same price, per 1M classification requests. (Self-hosting the fine-tuned model costs GPU time instead.)"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_MODELS = ["gpt-4o-mini", "claude-haiku-4-5", "gemini-2.5-flash", "deepseek-chat"]


def estimate_cost(tokens_in: float, tokens_out: float, in_price_per_m: float, out_price_per_m: float,
                  requests: int = 1_000_000) -> float:
    return requests * (tokens_in * in_price_per_m + tokens_out * out_price_per_m) / 1e6


def main(out_tokens: float = 8) -> None:
    base = json.loads((HERE / "results/baseline.json").read_text())
    ft = json.loads((HERE / "results/finetuned.json").read_text())
    with open(HERE / "data/llm_models.csv", newline="", encoding="utf-8") as f:
        prices = {r["model"]: r for r in csv.DictReader(f)}
    print(f"avg prompt tokens: baseline={base['avg_prompt_tokens']}  fine-tuned={ft['avg_prompt_tokens']}  "
          f"({base['avg_prompt_tokens'] / ft['avg_prompt_tokens']:.1f}x fewer)")
    print(f"accuracy: baseline={base['accuracy']}  fine-tuned={ft['accuracy']}\n")
    print(f"{'API model':<22}{'$/1M reqs baseline':>20}{'$/1M reqs short prompt':>24}")
    for m in DEFAULT_MODELS:
        if m in prices:
            p = prices[m]
            args = (float(p["input_usd_per_1m_tokens"]), float(p["output_usd_per_1m_tokens"]))
            print(f"{m:<22}{estimate_cost(base['avg_prompt_tokens'], out_tokens, *args):>20,.2f}"
                  f"{estimate_cost(ft['avg_prompt_tokens'], out_tokens, *args):>24,.2f}")


if __name__ == "__main__":
    main()
