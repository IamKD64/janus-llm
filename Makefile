CFG ?= configs/default.yaml
PY  ?= python

.PHONY: install install-cpu lint test baseline train eval-ft report gate promote all smoke ui serve

install:       ## GPU machine / Colab
	pip install -e ".[train,dev,serve]"
install-cpu:   ## laptop: tests + lint + report only
	pip install -e ".[dev,serve]"
lint:
	ruff check . && ruff format --check .
test:
	pytest -q --cov=p3ft --cov-report=term-missing

baseline:      ## prompt-engineering baselines
	$(PY) -m p3ft.evaluate --config $(CFG) --mode baseline
train:         ## QLoRA fine-tune
	$(PY) -m p3ft.train --config $(CFG)
eval-ft:       ## evaluate the adapter on the same held-out set
	$(PY) -m p3ft.evaluate --config $(CFG) --mode finetuned
report:
	$(PY) -m p3ft.report --config $(CFG)
gate:
	$(PY) -m p3ft.quality_gate --config $(CFG)
promote:       ## gate + register adapter in MLflow Model Registry
	$(PY) -m p3ft.quality_gate --config $(CFG) --promote

all: baseline train eval-ft report gate
smoke:
	$(MAKE) all CFG=configs/smoke.yaml

ui:
	mlflow ui --backend-store-uri $$($(PY) -c "import yaml;print(yaml.safe_load(open('$(CFG)'))['mlflow']['tracking_uri'])")
serve:
	uvicorn p3ft.serve:create_default_app --factory --port 8000
