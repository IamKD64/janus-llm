"""MLflow helpers."""

from __future__ import annotations

import os


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v if not isinstance(v, list) else ",".join(map(str, v))
    return out


def setup_mlflow(cfg: dict):
    import mlflow

    uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    # picked up by the Hugging Face Trainer's MLflow callback
    os.environ["MLFLOW_TRACKING_URI"] = uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg["mlflow"]["experiment"]
    return mlflow
