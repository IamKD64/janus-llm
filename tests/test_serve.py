from fastapi.testclient import TestClient

from p3ft.serve import create_app


class Fake:
    loaded = True

    def predict(self, text):
        return ("card_arrival", "card_arrival") if "card" in text else ("???", None)


def test_api():
    c = TestClient(create_app(Fake()))
    assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
    r = c.post("/classify", json={"text": "where is my card"}).json()
    assert r["label"] == "card_arrival" and r["latency_ms"] >= 0
    assert c.post("/classify", json={"text": "blah"}).json()["label"] is None
    assert c.post("/classify", json={"text": ""}).status_code == 422
