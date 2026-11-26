from p3ft.data import build_bundle, split_train_val, subsample


def test_subsample_deterministic(rows):
    assert subsample(rows, 10, 1) == subsample(rows, 10, 1)
    assert subsample(rows, 10, 1) != subsample(rows, 10, 2)
    assert subsample(rows, None, 1) == rows and len(subsample(rows, 999, 1)) == len(rows)


def test_split_disjoint(rows):
    train, val = split_train_val(rows, 8, 0)
    assert len(val) == 8 and len(train) == 32
    assert not {r["text"] for r in train} & {r["text"] for r in val}


def test_bundle_no_leakage(rows, labels):
    test = [{"text": f"test {i}", "label": labels[i % 4]} for i in range(20)]
    cfg = {"seed": 3, "data": {"train_size": 20, "val_size": 5, "eval_size": 10}}
    b = build_bundle(rows, test, labels, cfg)
    assert len(b.train) == 20 and len(b.val) == 5 and len(b.eval) == 10
    train_texts = {r["text"] for r in b.pool}
    assert not train_texts & {r["text"] for r in b.val}
    assert not train_texts & {r["text"] for r in b.eval}
