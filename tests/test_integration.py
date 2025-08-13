import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.parametrize("q", [
    "Summarize anomalies from August 1",
    "Why did RTT spike at 13:00 UTC yesterday?",
    "What was the average RTT in Los Angeles last week?",
    "List packet loss anomalies in Singapore region",
    "Give me a 24h summary of B-Root performance"
])
def test_ask(q):
    r = client.post("/ask", json={"query": q, "top_k": 20, "return_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["query"]
    # no guarantee of hits if empty DB, but endpoint should work
    assert "hits" in data

