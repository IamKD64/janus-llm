"""Release gate: the fine-tuned model must beat the BEST prompt baseline by a margin.

Exit code 1 on failure, so CI/CD can block merges and releases.
With --promote, a passing adapter is registered in the MLflow Model Registry
and given the `champion` alias.
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .report import best_baseline, finetuned, load_results


def evaluate_gate(results: list[dict], gate: dict) -> tuple[bool, list[str]]:
    base, ft = best_baseline(results), finetuned(results)
    if base is None or ft is None:
        return False, ["Need at least one baseline result AND a fine-tuned result."]
    bm, fm = base["metrics"], ft["metrics"]
    gain = fm["macro_f1"] - bm["macro_f1"]
    checks = [
        (
            gain >= gate["min_macro_f1_gain"],
            f"macro-F1 gain {gain:+.3f} vs {base['run_name']} (need >= {gate['min_macro_f1_gain']})",
        ),
        (
            fm["accuracy"] >= gate["min_accuracy"],
            f"accuracy {fm['accuracy']:.3f} (need >= {gate['min_accuracy']})",
        ),
        (
            fm["invalid_rate"] <= gate["max_invalid_rate"],
            f"invalid rate {fm['invalid_rate']:.3f} (need <= {gate['max_invalid_rate']})",
        ),
    ]
    return all(ok for ok, _ in checks), [
        f"[{'PASS' if ok else 'FAIL'}] {msg}" for ok, msg in checks
    ]


def promote(ft_result: dict, cfg: dict) -> str:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    run_id = ft_result.get("train_run_id")
    if not run_id:
        raise SystemExit("Fine-tuned result has no train_run_id; cannot register the adapter.")
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    client, name = MlflowClient(), cfg["mlflow"]["registered_model"]
    try:
        client.create_registered_model(name)
    except MlflowException:
        pass  # already exists
    mv = client.create_model_version(name, f"runs:/{run_id}/adapter", run_id=run_id)
    for k, v in ft_result["metrics"].items():
        client.set_model_version_tag(name, mv.version, k, f"{v:.4f}")
    client.set_registered_model_alias(name, "champion", mv.version)
    return f"{name} v{mv.version} -> alias 'champion'"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    results = load_results(cfg["results_dir"])
    ok, messages = evaluate_gate(results, cfg["quality_gate"])
    print("\n".join(messages))
    if not ok:
        print("QUALITY GATE: FAILED")
        sys.exit(1)
    print("QUALITY GATE: PASSED")
    if args.promote:
        print("Registered:", promote(finetuned(results), cfg))


if __name__ == "__main__":
    main()
