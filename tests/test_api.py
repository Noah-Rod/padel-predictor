"""The serving API contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from padel_predictor.common.features import FEATURE_COLUMNS
from padel_predictor.common.schemas import normalise_raw_matches
from padel_predictor.inference import api
from padel_predictor.inference.predict import Predictor
from padel_predictor.training.registry import ModelHandle


class StubModel:
    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p_a = 1.0 / (1.0 + np.exp(-x["elo_diff"].to_numpy() / 100.0))
        return np.column_stack([1.0 - p_a, p_a])


@pytest.fixture
def client(config, raw_matches, monkeypatch):
    handle = ModelHandle(
        model=StubModel(), name="padel-match-winner", version="3", alias="champion"
    )
    monkeypatch.setattr(api._State, "predictor", Predictor(handle, config))
    monkeypatch.setattr(api._State, "history", normalise_raw_matches(raw_matches))
    monkeypatch.setattr(api._State, "error", None)
    monkeypatch.setattr(api._State, "load", classmethod(lambda cls: None))
    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def unready_client(monkeypatch):
    monkeypatch.setattr(api._State, "predictor", None)
    monkeypatch.setattr(api._State, "history", None)
    monkeypatch.setattr(api._State, "error", "no model registered")
    monkeypatch.setattr(api._State, "load", classmethod(lambda cls: None))
    with TestClient(api.app) as test_client:
        yield test_client


def test_health_reports_the_deployed_model_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "3"


def test_health_is_degraded_but_answering_without_a_model(unready_client):
    """Observability first: an unready service must still explain itself."""
    response = unready_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["detail"] == "no model registered"


def test_prediction_endpoints_return_503_without_a_model(unready_client):
    assert unready_client.get("/upcoming").status_code == 503
    assert (
        unready_client.post(
            "/predict", json={"team_a": ["a", "b"], "team_b": ["c", "d"]}
        ).status_code
        == 503
    )


def test_predict_returns_a_probability_and_provenance(client):
    body = client.post("/predict", json={"team_a": ["P1", "P2"], "team_b": ["P3", "P4"]}).json()
    assert 0.0 <= body["probability_team_a"] <= 1.0
    assert body["predicted_winner"] in {"A", "B"}
    assert body["model_version"] == "3"


def test_predict_rejects_a_team_that_is_not_a_pair(client):
    response = client.post("/predict", json={"team_a": ["only-one"], "team_b": ["c", "d"]})
    assert response.status_code == 422


def test_predict_rejects_blank_player_names(client):
    response = client.post("/predict", json={"team_a": ["P1", "  "], "team_b": ["P3", "P4"]})
    assert response.status_code == 422


def test_players_endpoint_lists_the_history(client):
    players = client.get("/players").json()["players"]
    assert "P1" in players


def test_upcoming_returns_an_empty_list_when_nothing_is_scheduled(client):
    """The fixture history is fully played, so there is nothing to predict."""
    body = client.get("/upcoming").json()
    assert body["count"] == 0
    assert body["matches"] == []


def test_feature_order_is_the_contract():
    """The API and the model agree on feature order via one shared constant."""
    assert FEATURE_COLUMNS[0] == "elo_diff"
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
