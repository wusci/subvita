from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)


def test_list_runs_returns_200():
    r = client.get("/v1/runs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_get_run_404_for_unknown_id():
    r = client.get("/v1/runs/not-a-real-id")
    assert r.status_code == 404