"""Exercises the real MLflow tracking + model-registry path on a temp sqlite DB (no GPU needed)."""

import mlflow
from conftest import make_result
from mlflow.tracking import MlflowClient

from p3ft.quality_gate import promote


def test_promote_registers_adapter_with_champion_alias(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path}/m.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("t")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    with mlflow.start_run() as run:
        mlflow.log_artifacts(str(adapter), "adapter")

    cfg = {"mlflow": {"tracking_uri": uri, "registered_model": "demo-adapter"}}
    ft = make_result("finetuned-qlora", "finetuned", 0.9, train_run_id=run.info.run_id)
    msg = promote(ft, cfg)
    assert "champion" in msg

    client = MlflowClient()
    mv = client.get_model_version_by_alias("demo-adapter", "champion")
    assert mv.run_id == run.info.run_id
    assert mv.tags["macro_f1"] == "0.9000"
