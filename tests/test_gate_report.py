import json

from conftest import make_result

from p3ft import report
from p3ft.quality_gate import evaluate_gate

GATE = {"min_macro_f1_gain": 0.05, "min_accuracy": 0.8, "max_invalid_rate": 0.02}


def test_gate_pass():
    res = [
        make_result("zero_shot", "baseline", 0.55),
        make_result("one_per_label", "baseline", 0.70),
        make_result("finetuned-qlora", "finetuned", 0.90),
    ]
    ok, msgs = evaluate_gate(res, GATE)
    assert ok and all("PASS" in m for m in msgs)


def test_gate_fails_when_gain_too_small():
    res = [make_result("one_per_label", "baseline", 0.88), make_result("ft", "finetuned", 0.90)]
    ok, msgs = evaluate_gate(res, GATE)
    assert not ok and "FAIL" in msgs[0]


def test_gate_fails_on_invalid_rate_and_missing_runs():
    res = [make_result("b", "baseline", 0.5), make_result("ft", "finetuned", 0.9, inv=0.1)]
    assert not evaluate_gate(res, GATE)[0]
    assert not evaluate_gate([make_result("b", "baseline", 0.5)], GATE)[0]


def test_report_markdown_and_plot(tmp_path):
    res = [
        make_result("zero_shot", "baseline", 0.5),
        make_result("finetuned-qlora", "finetuned", 0.9),
    ]
    for r in res:
        (tmp_path / f"{r['run_name']}.json").write_text(json.dumps(r))
    loaded = report.load_results(tmp_path)
    assert loaded[0]["run_name"] == "finetuned-qlora"  # sorted by macro-F1
    md = report.render_markdown(loaded)
    assert "+0.400" in md and "finetuned-qlora" in md
    report.plot(loaded, tmp_path / "c.png")
    assert (tmp_path / "c.png").stat().st_size > 0
