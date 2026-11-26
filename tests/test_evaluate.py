from p3ft.evaluate import run_eval
from p3ft.modeling import GenOutput
from p3ft.prompts import PromptBuilder


def test_run_eval_with_mock_generator(rows, labels):
    builder = PromptBuilder("zero_shot", labels)
    lookup = {r["text"]: r["label"] for r in rows}
    calls = []

    def perfect(batch_msgs):
        calls.append(len(batch_msgs))
        return [GenOutput(lookup[m[-1]["content"]], 50) for m in batch_msgs]

    res = run_eval(rows, labels, builder, perfect, batch_size=16)
    assert calls == [16, 16, 8]
    m = res["metrics"]
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["avg_prompt_tokens"] == 50
    assert res["top_confusions"] == [] and len(res["predictions"]) == 40


def test_run_eval_records_errors(rows, labels):
    builder = PromptBuilder("zero_shot", labels)
    res = run_eval(
        rows, labels, builder, lambda b: [GenOutput("exchange_rate", 10) for _ in b], batch_size=8
    )
    assert res["metrics"]["accuracy"] == 0.25
    assert res["top_confusions"][0]["pred"] == "exchange_rate"


def test_save_and_log_writes_json_and_mlflow(tmp_path, rows, labels):
    import json

    import mlflow

    from p3ft.evaluate import save_and_log

    uri = f"sqlite:///{tmp_path}/m.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("t")
    cfg = {
        "results_dir": str(tmp_path / "res"),
        "model": {"name": "m"},
        "seed": 1,
        "eval": {"batch_size": 8, "strategies": ["zero_shot"]},
    }
    res = run_eval(
        rows,
        labels,
        PromptBuilder("zero_shot", labels),
        lambda b: [GenOutput("card_arrival", 5) for _ in b],
        8,
    )
    summary = save_and_log(mlflow, cfg, "baseline-zero_shot", "baseline", "zero_shot", res, {})
    saved = json.loads((tmp_path / "res" / "baseline-zero_shot.json").read_text())
    assert saved["mlflow_run_id"] == summary["mlflow_run_id"]
    run = mlflow.get_run(summary["mlflow_run_id"])
    assert run.data.metrics["accuracy"] == 0.25
    assert run.data.params["strategy"] == "zero_shot"
