import csv
from pathlib import Path

DATA = Path(__file__).parent / "data"


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{"text": r["text"], "label": r["category"]} for r in csv.DictReader(f)]


def load_banking77():
    """Returns (train, test, label_names) from the bundled REAL Banking77 CSVs
    (PolyAI, CC-BY-4.0: 10,003 train / 3,080 test customer-service queries, 77 intents). No internet needed."""
    train, test = _read(DATA / "banking77_train.csv"), _read(DATA / "banking77_test.csv")
    labels = sorted({r["label"] for r in train})
    return train, test, labels
