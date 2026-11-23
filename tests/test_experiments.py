"""Experiments on the training pipeline's split: ablations and training windows."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from padel_predictor.common import storage
from padel_predictor.common.features import FEATURE_COLUMNS, build_feature_frame
from padel_predictor.training import experiments, pipeline
from padel_predictor.training.dataset import time_split


@pytest.fixture
def stored_features(settings, config, many_matches):
    features = build_feature_frame(many_matches, config.features)
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    return features


@pytest.fixture
def window_config(config):
    """The synthetic history is about five months of training matches."""
    config = config.model_copy(deep=True)
    config.training.training_window.months = [12, 3, 1]
    return config


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
def test_each_configured_feature_is_dropped_in_turn(settings, config, stored_features):
    report = experiments.run_ablation(settings=settings, config=config, track=False)

    assert [v.dropped for v in report.variants] == [None, *config.training.ablation.drop]
    for variant in report.variants[1:]:
        assert variant.dropped not in variant.features
        assert len(variant.features) == len(FEATURE_COLUMNS) - 1
        gap = variant.vs_reference
        assert gap["ci_low"] <= gap["mean"] <= gap["ci_high"]
    assert report.variants[0].features == FEATURE_COLUMNS
    assert report.variants[0].vs_reference is None


def test_the_reference_variant_is_the_training_pipelines_model(settings, config, stored_features):
    report = experiments.run_ablation(settings=settings, config=config, track=False)
    trained = pipeline.run(settings=settings, config=config, register=False)

    assert report.variants[0].metrics == pytest.approx(trained.metrics)
    assert report.variants[0].train_rows == trained.train_rows
    assert report.test_rows == trained.test_rows


def test_every_variant_is_tracked_and_none_is_registered(settings, config, stored_features):
    experiments.run_ablation(settings=settings, config=config, track=True)

    root = Path(settings.local_registry_dir)
    records = [json.loads(p.read_text()) for p in (root / "runs").glob("*.json")]
    assert sorted(r["name"] for r in records) == sorted(
        ["ablation-none-all-features"]
        + [f"ablation-{f.replace('_', '-')}" for f in config.training.ablation.drop]
    )
    assert all(r["tags"]["experiment"] == "ablation" for r in records)
    assert not (root / settings.registered_model_name).exists()


def test_the_report_has_one_row_per_variant(settings, config, stored_features):
    markdown = experiments.run_ablation(settings=settings, config=config, track=False)
    lines = markdown.to_markdown().splitlines()

    rows = [line for line in lines if line.startswith("| ")]
    # Header and separator, the reference model, one row per dropped feature.
    assert len(rows) == 2 + 1 + len(config.training.ablation.drop)
    assert any("partnership_diff" in row for row in rows)
    # No provider rating in this store: the rated-only gap is shown as missing.
    assert not any("nan" in line for line in lines)


def test_an_unknown_feature_is_rejected(settings, config, stored_features):
    config = config.model_copy(deep=True)
    config.training.ablation.drop = ["provider_elo_prob"]
    with pytest.raises(ValueError, match="cannot drop"):
        experiments.run_ablation(settings=settings, config=config, track=False)


# --------------------------------------------------------------------------- #
# Training window
# --------------------------------------------------------------------------- #
def test_a_window_fits_only_on_its_last_months(settings, window_config, stored_features):
    report = experiments.run_training_window(settings=settings, config=window_config, track=False)

    split = time_split(stored_features, window_config.training.test_fraction)
    played = pd.to_datetime(split.meta_train["played_at"], utc=True)
    since = split.train_end - pd.DateOffset(months=3)

    reference, window = report.variants
    assert (reference.label, window.label) == ("all history", "last 3 months")
    assert reference.train_rows == len(split.x_train)
    assert window.train_rows == int((played >= since).sum()) < reference.train_rows
    assert pd.Timestamp(window.train_start, tz="UTC") >= since.normalize()
    # Same test matches for every variant, compared with the model on all history.
    assert window.vs_reference["ci_low"] <= window.vs_reference["ci_high"]


def test_a_window_works_on_timestamps_without_a_time_zone(settings, window_config, many_matches):
    """Hopsworks returns ``played_at`` without a time zone, the Parquet store with
    UTC. The first run against the live store failed on comparing the two."""
    features = build_feature_frame(many_matches, window_config.features)
    aware = experiments_rows(settings, window_config, features)
    features["played_at"] = features["played_at"].dt.tz_localize(None)
    naive = experiments_rows(settings, window_config, features)

    assert naive == aware


def experiments_rows(settings, config, features):
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    report = experiments.run_training_window(settings=settings, config=config, track=False)
    return [(v.label, v.train_rows) for v in report.variants]


def test_windows_that_change_nothing_or_hold_too_little_are_skipped(
    settings, window_config, stored_features
):
    report = experiments.run_training_window(settings=settings, config=window_config, track=False)

    assert [v.months for v in report.variants] == [None, 3]
    assert report.skipped[0] == "12 months (covers all history)"
    assert report.skipped[1].startswith("1 months (only ")
    assert "Skipped: 12 months" in report.to_markdown()


def test_window_runs_are_tracked_under_their_own_experiment(
    settings, window_config, stored_features
):
    experiments.run_training_window(settings=settings, config=window_config, track=True)

    root = Path(settings.local_registry_dir)
    records = [json.loads(p.read_text()) for p in (root / "runs").glob("*.json")]
    assert sorted(r["name"] for r in records) == [
        "training-window-all-history",
        "training-window-last-3-months",
    ]
    assert {r["tags"]["experiment"] for r in records} == {"training-window"}
    assert not (root / settings.registered_model_name).exists()


def test_a_window_must_be_positive(settings, config, stored_features):
    config = config.model_copy(deep=True)
    config.training.training_window.months = [0]
    with pytest.raises(ValueError, match="positive"):
        experiments.run_training_window(settings=settings, config=config, track=False)


# --------------------------------------------------------------------------- #
# Elo K-factor
# --------------------------------------------------------------------------- #
@pytest.fixture
def k_config(config):
    config = config.model_copy(deep=True)
    config.features.elo_k_factor = 24.0
    config.training.elo_k.factors = [16, 32]
    config.training.elo_k.validation_folds = 2
    return config


@pytest.fixture
def stored_matches(settings, many_matches):
    storage.write_snapshot(many_matches, settings.feature_store_uri, storage.RAW_MATCHES)
    return many_matches


def test_each_k_factor_is_a_variant_next_to_the_current_one(settings, k_config, stored_matches):
    report = experiments.run_elo_k(settings=settings, config=k_config, track=False)

    # The K in features.elo_k_factor is the reference, whether listed or not.
    assert [v.elo_k for v in report.variants] == [24.0, 16.0, 32.0]
    assert report.variants[0].vs_reference is None
    for variant in report.variants[1:]:
        gap = variant.vs_reference
        assert gap["ci_low"] <= gap["mean"] <= gap["ci_high"]
    # Same matches on each side of the split for every K.
    assert len({v.train_rows for v in report.variants}) == 1


def test_the_k_factor_is_chosen_on_the_validation_folds(settings, k_config, stored_matches):
    report = experiments.run_elo_k(settings=settings, config=k_config, track=False)

    best = min(report.variants, key=lambda v: v.validation_log_loss)
    assert [v.selected for v in report.variants].count(True) == 1
    assert best.selected
    markdown = report.to_markdown()
    assert markdown.count("(chosen)") == 1
    verdict = markdown.splitlines()[-1]
    assert f"K = {best.elo_k:g}" in verdict
    # A K other than the current one comes with its interval against the current one.
    assert best.elo_k == 24.0 or "against K = 24 now" in verdict
    # No provider rating in this store: the in-house Elo on rated matches is missing.
    assert "nan" not in markdown


def test_the_validation_folds_never_read_the_test_set(settings, k_config, stored_matches):
    features = build_feature_frame(stored_matches, k_config.features)
    split = time_split(features, k_config.training.test_fraction)
    flipped = dataclasses.replace(split, y_test=1 - split.y_test, x_test=split.x_test * 0.0)

    y, proba = experiments.validation_predictions(split, k_config, 2)
    y_flipped, proba_flipped = experiments.validation_predictions(flipped, k_config, 2)

    assert y.equals(y_flipped)
    assert proba == pytest.approx(proba_flipped)
    # Every fold is predicted from earlier matches only: the first block is never scored.
    assert 0 < len(y) < len(split.y_train)
    assert y.tolist() == split.y_train.iloc[-len(y) :].tolist()
    with pytest.raises(ValueError, match="at least 1"):
        experiments.validation_predictions(split, k_config, 0)


def test_the_current_k_factor_reproduces_the_training_pipelines_model(
    settings, k_config, stored_matches
):
    features = build_feature_frame(stored_matches, k_config.features)
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)

    report = experiments.run_elo_k(settings=settings, config=k_config, track=False)
    trained = pipeline.run(settings=settings, config=k_config, register=False)

    assert report.variants[0].metrics == pytest.approx(trained.metrics)
    assert report.variants[0].train_rows == trained.train_rows


def test_k_factor_runs_are_tracked_under_their_own_experiment(settings, k_config, stored_matches):
    experiments.run_elo_k(settings=settings, config=k_config, track=True)

    root = Path(settings.local_registry_dir)
    records = [json.loads(p.read_text()) for p in (root / "runs").glob("*.json")]
    assert sorted(r["name"] for r in records) == ["elo-k-k-16", "elo-k-k-24", "elo-k-k-32"]
    assert {r["tags"]["experiment"] for r in records} == {"elo-k"}
    assert sorted(r["tags"]["chosen_on_validation"] for r in records) == ["False", "False", "True"]
    assert all("validation_log_loss" in r["metrics"] for r in records)
    assert not (root / settings.registered_model_name).exists()


def test_a_k_factor_must_be_positive_and_the_store_must_hold_matches(settings, k_config):
    with pytest.raises(RuntimeError, match="no raw matches"):
        experiments.run_elo_k(settings=settings, config=k_config, track=False)

    k_config.training.elo_k.factors = [0]
    with pytest.raises(ValueError, match="positive"):
        experiments.run_elo_k(settings=settings, config=k_config, track=False)
