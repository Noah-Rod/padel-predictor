"""The Streamlit page, run headless against a store filled by the real pipelines."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from padel_predictor.common import storage
from padel_predictor.common.config import get_pipeline_config, get_settings
from padel_predictor.common.features import build_feature_frame, labelled
from padel_predictor.features import pipeline as feature_pipeline
from padel_predictor.inference.predict import Predictor, load_history
from padel_predictor.training import pipeline as training_pipeline

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


def run_app() -> AppTest:
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.fixture
def live_store():
    """History, fixtures and a champion, as a fresh clone has after the README steps."""
    today = dt.datetime.now(dt.UTC).date()
    feature_pipeline.backfill(today - dt.timedelta(days=300), today - dt.timedelta(days=1))
    feature_pipeline.run()
    training_pipeline.run()


def test_page_explains_itself_without_a_model():
    app = run_app()
    assert not app.exception
    assert "not ready" in app.error[0].value


def test_upcoming_view_lists_the_scheduled_matches(live_store):
    app = run_app()
    assert not app.exception
    assert app.segmented_control(key="view").value == "Upcoming matches"
    upcoming = int(app.metric[2].value.replace(",", ""))
    assert upcoming > 0
    assert len(app.dataframe[0].value) == upcoming


def test_builder_starts_from_the_first_upcoming_match_and_reproduces_it(live_store):
    app = run_app()
    app.segmented_control(key="view").set_value("Matchup builder").run()
    assert not app.exception

    settings, config = get_settings(), get_pipeline_config()
    predictor = Predictor.load(settings, config)
    first = predictor.predict_upcoming(load_history(settings)).iloc[0]
    assert app.multiselect(key="team_a").value == [
        first["team_a_player_1"],
        first["team_a_player_2"],
    ]
    shown = [m.value for m in app.metric if "Team A ·" in m.label]
    assert shown == [f"{first['probability_team_a']:.1%}"]


def test_builder_refuses_a_player_on_both_teams(live_store):
    app = run_app()
    app.segmented_control(key="view").set_value("Matchup builder").run()
    team_a = app.multiselect(key="team_a").value
    app.multiselect(key="team_b").set_value([team_a[0], app.multiselect(key="team_b").value[1]])
    app.run()
    assert not app.exception
    assert "cannot play on both teams" in app.warning[0].value


def test_track_record_scores_the_stored_predictions(live_store):
    settings, config = get_settings(), get_pipeline_config()
    predictor = Predictor.load(settings, config)
    finished = labelled(
        build_feature_frame(load_history(settings), config.features, rankings=predictor.rankings)
    ).tail(40)
    predictions = predictor.predict_frame(finished)
    predictions["predicted_at"] = predictions["played_at"] - pd.Timedelta(hours=12)
    storage.write_snapshot(predictions, settings.feature_store_uri, storage.PREDICTIONS)

    app = run_app()
    app.segmented_control(key="view").set_value("Track record").run()
    assert not app.exception
    assert [m.value for m in app.metric if m.label == "Matches scored"] == ["40"]


def test_track_record_says_so_when_nothing_is_scored_yet(live_store):
    app = run_app()
    app.segmented_control(key="view").set_value("Track record").run()
    assert not app.exception
    assert "No stored prediction has a result yet" in app.info[0].value
