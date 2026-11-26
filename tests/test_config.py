import pytest

from p3ft.config import load_config


def test_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: {b: 1}\nname: x\n")
    cfg = load_config(p, ["a.b=5", "a.c=[1,2]", "name=y", "new.deep=true"])
    assert cfg["a"] == {"b": 5, "c": [1, 2]}
    assert cfg["name"] == "y" and cfg["new"]["deep"] is True


def test_bad_override(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: 1\n")
    with pytest.raises(ValueError):
        load_config(p, ["oops"])


def test_shipped_configs_load():
    for name in ("default", "smoke"):
        cfg = load_config(f"configs/{name}.yaml")
        assert {"data", "model", "qlora", "train", "eval", "mlflow", "quality_gate"} <= set(cfg)
