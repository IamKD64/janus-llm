"""Build results/REPORT.md and results/comparison.png from results/*.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def load_results(results_dir: str | Path) -> list[dict]:
    rows = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]
    return sorted(rows, key=lambda r: r["metrics"]["macro_f1"], reverse=True)


def best_baseline(results: list[dict]) -> dict | None:
    base = [r for r in results if r["mode"] == "baseline"]
    return max(base, key=lambda r: r["metrics"]["macro_f1"]) if base else None


def finetuned(results: list[dict]) -> dict | None:
    ft = [r for r in results if r["mode"] == "finetuned"]
    return max(ft, key=lambda r: r["timestamp"]) if ft else None


def render_markdown(results: list[dict]) -> str:
    lines = [
        "# Results: QLoRA fine-tune vs prompt engineering",
        "",
        f"Model: `{results[0]['model']}` | eval set: {results[0]['eval_size']} held-out "
        f"Banking77 test examples | seed {results[0]['seed']}",
        "",
        "| Run | Accuracy (95% CI) | Macro-F1 | Invalid % | Prompt tokens | ms/sample |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r["metrics"]
        lines.append(
            f"| {r['run_name']} | {m['accuracy']:.3f} "
            f"({m['accuracy_ci_low']:.3f}-{m['accuracy_ci_high']:.3f}) | {m['macro_f1']:.3f} | "
            f"{100 * m['invalid_rate']:.1f} | {m['avg_prompt_tokens']:.0f} | "
            f"{m['avg_latency_ms']:.0f} |"
        )
    base, ft = best_baseline(results), finetuned(results)
    if base and ft:
        d = ft["metrics"]["macro_f1"] - base["metrics"]["macro_f1"]
        tok_ratio = base["metrics"]["avg_prompt_tokens"] / max(
            ft["metrics"]["avg_prompt_tokens"], 1
        )
        lines += [
            "",
            "## Takeaways",
            f"- Fine-tuned macro-F1 is **{d:+.3f}** vs the best prompt baseline (`{base['run_name']}`).",
            f"- The fine-tuned prompt is {tok_ratio:.1f}x shorter "
            f"({ft['metrics']['avg_prompt_tokens']:.0f} vs {base['metrics']['avg_prompt_tokens']:.0f} "
            "tokens), which is where the inference-cost savings come from.",
        ]
        if ft.get("top_confusions"):
            lines += [
                "",
                "## Most common fine-tuned errors",
                "",
                "| True | Predicted | Count |",
                "|---|---|---|",
            ]
            lines += [
                f"| {c['true']} | {c['pred']} | {c['count']} |" for c in ft["top_confusions"][:5]
            ]
    return "\n".join(lines) + "\n"


def plot(results: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["run_name"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = range(len(names))
    ax.bar(
        [x - 0.2 for x in xs], [r["metrics"]["accuracy"] for r in results], 0.4, label="accuracy"
    )
    ax.bar(
        [x + 0.2 for x in xs], [r["metrics"]["macro_f1"] for r in results], 0.4, label="macro-F1"
    )
    ax.set_xticks(list(xs), names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("QLoRA fine-tune vs prompt baselines (Banking77)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    results = load_results(cfg["results_dir"])
    if not results:
        raise SystemExit(f"No result JSON files in {cfg['results_dir']}/ - run evaluation first.")
    out = Path(cfg["results_dir"])
    (out / "REPORT.md").write_text(render_markdown(results))
    plot(results, out / "comparison.png")
    print(f"Wrote {out / 'REPORT.md'} and {out / 'comparison.png'}")


if __name__ == "__main__":
    main()
