"""Ablations: each feature dropped in turn, on the training pipeline's split."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from padel_predictor.common import storage
from padel_predictor.common.features import FEATURE_COLUMNS, build_feature_frame
from padel_predictor.training import ablation, pipeline


@pytest.fixture
def stored_features(settings, config, many_matches):
    features = build_feature_frame(many_matches, config.features)
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    return features


def test_each_configured_feature_is_dropped_in_turn(settings, config, stored_features):
    report = ablation.run(settings=settings, config=config, track=False)

    assert [r.dropped for r in report.results] == [None, *config.training.ablation.drop]
    for result in report.results[1:]:
        assert result.dropped not in result.features
        assert len(result.features) == len(FEATURE_COLUMNS) - 1
        gap = result.vs_all_features
        assert gap["ci_low"] <= gap["mean"] <= gap["ci_high"]
    assert report.results[0].features == FEATURE_COLUMNS
    assert report.results[0].vs_all_features is None


def test_the_all_feature_variant_is_the_training_pipelines_model(settings, config, stored_features):
    report = ablation.run(settings=settings, config=config, track=False)
    trained = pipeline.run(settings=settings, config=config, register=False)

    assert report.results[0].metrics == pytest.approx(trained.metrics)
    assert report.test_rows == trained.test_rows


def test_every_variant_is_tracked_and_none_is_registered(settings, config, stored_features):
    ablation.run(settings=settings, config=config, track=True)

    root = Path(settings.local_registry_dir)
    records = [json.loads(p.read_text()) for p in (root / "runs").glob("*.json")]
    assert sorted(r["name"] for r in records) == sorted(
        ["ablation-all-features"] + [f"ablation-no-{f}" for f in config.training.ablation.drop]
    )
    assert all(r["tags"]["ablation"] == "True" for r in records)
    assert not (root / settings.registered_model_name).exists()


def test_the_report_has_one_row_per_variant(settings, config, stored_features):
    markdown = ablation.run(settings=settings, config=config, track=False).to_markdown()

    rows = [line for line in markdown.splitlines() if line.startswith("| ")]
    # Header and separator, the all-feature model, one row per dropped feature.
    assert len(rows) == 2 + 1 + len(config.training.ablation.drop)
    assert "partnership_diff" in markdown
    # No provider rating in this store: the rated-only gap is shown as missing.
    assert "nan" not in markdown


def test_an_unknown_feature_is_rejected(settings, config, stored_features):
    config = config.model_copy(deep=True)
    config.training.ablation.drop = ["provider_elo_prob"]
    with pytest.raises(ValueError, match="cannot drop"):
        ablation.run(settings=settings, config=config, track=False)
