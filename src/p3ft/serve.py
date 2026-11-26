"""FastAPI inference service for the fine-tuned adapter.

  ADAPTER_PATH=outputs/qlora-banking77 MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
  uvicorn p3ft.serve:create_default_app --factory --port 8000
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .metrics import normalize_label


class Classifier(Protocol):
    loaded: bool

    def predict(self, text: str) -> tuple[str, str | None]:
        """Returns (raw_generation, normalised_label_or_None)."""


class LlmClassifier:
    """Loads the 4-bit base model + adapter lazily on first request."""

    def __init__(self, model_name: str, adapter_path: str):
        self.model_name, self.adapter_path = model_name, adapter_path
        self.loaded = False
        self._lock = threading.Lock()

    def _load(self) -> None:
        from .modeling import attach_adapter, load_base_model, load_tokenizer, make_generate_fn
        from .prompts import PromptBuilder

        self.labels = json.loads((Path(self.adapter_path) / "labels.json").read_text())
        self.builder = PromptBuilder("finetuned", self.labels)
        tok = load_tokenizer(self.model_name)
        model = attach_adapter(load_base_model(self.model_name, quantize=True), self.adapter_path)
        self._generate = make_generate_fn(model, tok, max_new_tokens=16)
        self.loaded = True

    def predict(self, text: str) -> tuple[str, str | None]:
        with self._lock:
            if not self.loaded:
                self._load()
            raw = self._generate([self.builder.messages(text)])[0].text
        return raw, normalize_label(raw, self.labels)


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ClassifyResponse(BaseModel):
    label: str | None
    raw: str
    latency_ms: float


def create_app(classifier: Classifier) -> FastAPI:
    app = FastAPI(title="Banking77 intent classifier (QLoRA)", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": classifier.loaded}

    @app.post("/classify", response_model=ClassifyResponse)
    def classify(req: ClassifyRequest) -> ClassifyResponse:
        t0 = time.perf_counter()
        raw, label = classifier.predict(req.text)
        return ClassifyResponse(label=label, raw=raw, latency_ms=1000 * (time.perf_counter() - t0))

    return app


def create_default_app() -> FastAPI:
    return create_app(
        LlmClassifier(
            os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            os.environ.get("ADAPTER_PATH", "outputs/qlora-banking77"),
        )
    )
