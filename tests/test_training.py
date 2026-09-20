"""Splitting, scoring and the promotion decision.

MLflow is deliberately not exercised here: these tests cover the logic that
decides *whether* a model ships, which is the part that can silently go wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from padel_predictor.common.features import build_feature_frame
from padel_predictor.training import pipeline, registry
from padel_predictor.training.dataset import (
    baseline_metrics,
    classification_metrics,
    time_split,
)


@pytest.fixture
def features(many_matches, feature_config):
    return build_feature_frame(many_matches, feature_config)


def test_split_is_chronological(features):
    split = time_split(features, test_fraction=0.2)

    assert split.train_end <= split.test_start
    labelled_rows = len(features.dropna(subset=["team_a_won"]))
    assert len(split.x_test) == pytest.approx(labelled_rows * 0.2, rel=0.1)
    assert not split.x_train.empty


def test_split_rejects_an_impossible_fraction(features):
    with pytest.raises(ValueError, match="test_fraction"):
        time_split(features, test_fraction=1.5)


def test_split_needs_labelled_rows(features):
    unlabelled_only = features.assign(team_a_won=None)
    with pytest.raises(ValueError, match="no labelled feature rows"):
        time_split(unlabelled_only)


def test_metrics_reward_a_confident_correct_prediction():
    good = classification_metrics([1, 0, 1, 0], np.array([0.9, 0.1, 0.8, 0.2]))
    bad = classification_metrics([1, 0, 1, 0], np.array([0.1, 0.9, 0.2, 0.8]))

    assert good["accuracy"] == 1.0
    assert good["log_loss"] < bad["log_loss"]
    assert good["brier"] < bad["brier"]


def test_roc_auc_is_nan_for_a_single_class_holdout():
    metrics = classification_metrics([1, 1, 1], np.array([0.6, 0.7, 0.8]))
    assert np.isnan(metrics["roc_auc"])


def test_baselines_are_reported_for_majority_and_elo(features):
    baselines = baseline_metrics(time_split(features, 0.2))
    assert "baseline_majority_log_loss" in baselines
    assert "baseline_elo_log_loss" in baselines
    assert 0.0 <= baselines["baseline_elo_accuracy"] <= 1.0


def test_first_model_is_always_promoted():
    promote, reason = registry.should_promote(
        {"log_loss": 0.9}, None, "log_loss", higher_is_better=False, min_improvement=0.002
    )
    assert promote
    assert "no champion" in reason


def test_a_better_model_is_promoted():
    promote, reason = registry.should_promote(
        {"log_loss": 0.60},
        {"log_loss": 0.65},
        "log_loss",
        higher_is_better=False,
        min_improvement=0.002,
    )
    assert promote
    assert "improved" in reason


def test_a_marginally_better_model_is_not_promoted():
    """Noise-sized gains must not churn the deployed model."""
    promote, _ = registry.should_promote(
        {"log_loss": 0.6499},
        {"log_loss": 0.6500},
        "log_loss",
        higher_is_better=False,
        min_improvement=0.002,
    )
    assert not promote


def test_a_worse_model_is_not_promoted():
    promote, reason = registry.should_promote(
        {"log_loss": 0.80},
        {"log_loss": 0.65},
        "log_loss",
        higher_is_better=False,
        min_improvement=0.002,
    )
    assert not promote
    assert "did not improve" in reason


def test_higher_is_better_metrics_compare_the_other_way():
    promote, _ = registry.should_promote(
        {"accuracy": 0.70},
        {"accuracy": 0.65},
        "accuracy",
        higher_is_better=True,
        min_improvement=0.01,
    )
    assert promote


def test_training_runs_and_beats_the_majority_baseline(settings, config, many_matches):
    """An end-to-end train with registration disabled: no MLflow, no registry."""
    from padel_predictor.common import storage

    features = build_feature_frame(many_matches, config.features)
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)

    result = pipeline.run(settings=settings, config=config, register=False)

    assert result.train_rows > 0
    assert result.test_rows > 0
    assert not result.promoted
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.metrics["log_loss"] < result.baselines["baseline_majority_log_loss"]


def test_training_refuses_to_run_on_too_little_data(settings, config, raw_matches):
    from padel_predictor.common import storage

    features = build_feature_frame(raw_matches, config.features)
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)

    with pytest.raises(RuntimeError, match="need at least"):
        pipeline.run(settings=settings, config=config, register=False)


def test_training_without_a_feature_store_fails_loudly(settings, config):
    with pytest.raises(RuntimeError, match="no feature snapshots"):
        pipeline.run(settings=settings, config=config, register=False)
