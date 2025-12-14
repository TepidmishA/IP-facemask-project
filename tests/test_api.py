import os

import cv2
import numpy as np
from fastapi.testclient import TestClient

# Use lightweight dummy models to avoid loading real weights during tests
os.environ["USE_DUMMY_MODELS"] = "1"

from server.app import app  # noqa: E402


def test_predict_endpoint_returns_schema():
    client = TestClient(app)
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", img)
    response = client.post("/predict?model=dl", files={"image": ("test.png", buf.tobytes(), "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "model" in data
    assert "details" in data

