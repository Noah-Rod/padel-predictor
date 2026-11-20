"""Splitting, scoring and the promotion decision.

These tests cover the logic that decides *whether* a model ships, which is the
part that can silently go wrong, and the registry contract through the local
backend: versions, the champion alias and loading it back.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from padel_predictor.common.features import build_feature_frame
from padel_predictor.training import pipeline, registry
from padel_predictor.training.dataset import (
    baseline_metrics,
    bootstrap_log_loss_difference,
    classification_metrics,
    provider_elo_probabilities,
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


def test_the_model_trains_on_a_feature_that_is_missing_in_some_rows(features, config):
    """Missing ranking points reach the classifier as they are (review point A3)."""
    split = time_split(features, 0.2)
    x_train = split.x_train.copy()
    x_train["rank_points_diff"] = x_train["elo_diff"]
    x_train.loc[x_train.index[::3], "rank_points_diff"] = np.nan

    model = pipeline.build_model(config).fit(x_train, split.y_train)

    assert model.named_steps["empty_columns"].empty_columns_ == []
    passed_on = model.named_steps["empty_columns"].transform(x_train)
    assert passed_on["rank_points_diff"].isna().sum() == x_train["rank_points_diff"].isna().sum()
    proba = model.predict_proba(split.x_test.assign(rank_points_diff=np.nan))[:, 1]
    assert np.isfinite(proba).all()


def test_the_model_trains_without_any_ranking_points(features, config):
    """No ranking history at all, as on the free plan: the column is ignored."""
    split = time_split(features, 0.2)
    x_train = split.x_train.assign(rank_points_diff=np.nan)

    model = pipeline.build_model(config).fit(x_train, split.y_train)

    assert model.named_steps["empty_columns"].empty_columns_ == ["rank_points_diff"]
    with_points = model.predict_proba(split.x_test.assign(rank_points_diff=500.0))[:, 1]
    without = model.predict_proba(split.x_test.assign(rank_points_diff=np.nan))[:, 1]
    np.testing.assert_allclose(with_points, without)


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
    """An end-to-end train with registration disabled: no tracker, no registry."""
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


def test_split_never_cuts_through_a_tournament(features):
    split = time_split(features, test_fraction=0.2)
    train_tournaments = set(features.loc[features["played_at"] <= split.train_end, "tournament"])
    assert not train_tournaments & set(split.meta_test["tournament"])


def test_provider_baseline_is_scored_on_every_test_row(features):
    split = time_split(features, 0.2)
    meta = split.meta_test.assign(provider_elo_prob=0.7)
    meta.loc[meta.index[:10], "provider_elo_prob"] = float("nan")
    split = dataclasses.replace(split, meta_test=meta)

    proba, coverage = provider_elo_probabilities(split)
    assert len(proba) == len(split.y_test)
    assert (proba[:10] == 0.5).all()
    assert coverage == pytest.approx(1 - 10 / len(split.y_test))


def test_baselines_include_the_provider_elo(features):
    baselines = baseline_metrics(time_split(features, 0.2))
    assert "baseline_provider_elo_log_loss" in baselines
    assert 0.0 <= baselines["baseline_provider_elo_coverage"] <= 1.0


def test_bootstrap_does_not_call_an_identical_model_a_win():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = rng.uniform(0.2, 0.8, 2000)
    result = bootstrap_log_loss_difference(y, p, p, samples=500)
    assert result["mean"] == pytest.approx(0.0)
    assert result["beats_baseline"] == 0.0


def test_bootstrap_interval_contains_zero_for_a_tiny_noisy_edge():
    """~2,000 matches and a 0.001 log-loss gap: not distinguishable from noise."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    base = np.clip(np.where(y == 1, 0.6, 0.4) + rng.normal(0, 0.15, 2000), 0.05, 0.95)
    model = np.clip(base + rng.normal(0, 0.02, 2000), 0.05, 0.95)
    result = bootstrap_log_loss_difference(y, model, base, samples=1000)
    assert result["ci_low"] < 0 < result["ci_high"]
    assert result["beats_baseline"] == 0.0


def test_bootstrap_recognises_a_clearly_better_model():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    good = np.where(y == 1, 0.75, 0.25)
    result = bootstrap_log_loss_difference(y, good, np.full(2000, 0.5), samples=500)
    assert result["ci_high"] < 0
    assert result["beats_baseline"] == 1.0


def test_training_reports_the_gap_to_the_provider_elo(settings, config):
    from padel_predictor.features import pipeline as feature_pipeline

    end = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)
    feature_pipeline.backfill(end - dt.timedelta(days=200), end, settings=settings, config=config)
    result = pipeline.run(settings=settings, config=config, register=False)

    for key in (
        "vs_provider_elo_log_loss_diff",
        "vs_provider_elo_ci_low",
        "vs_provider_elo_ci_high",
    ):
        assert np.isfinite(result.baselines[key])
    assert result.baselines["vs_provider_elo_ci_low"] <= result.baselines["vs_provider_elo_ci_high"]


def test_the_run_summary_leads_with_the_rated_matches(settings, config):
    """The criterion is the gap where the provider has a rating (editorial point E1)."""
    from padel_predictor.features import pipeline as feature_pipeline
    from padel_predictor.features.report import build_report

    end = dt.datetime.now(dt.UTC).date() - dt.timedelta(days=1)
    feature_pipeline.backfill(end - dt.timedelta(days=200), end, settings=settings, config=config)
    result = pipeline.run(settings=settings, config=config, register=False)

    summary = result.summary()
    assert summary.index("rated ") < summary.index("all matches, unrated at 0.5")
    assert f"criterion met={result.success_criterion_met}" in summary
    assert result.success_criterion_met == bool(result.baselines["covered_beats_provider_elo"])

    report = build_report(settings=settings, config=config).to_markdown()
    rows = [line for line in report.splitlines() if "Elo" in line]
    assert rows[0].startswith("| **Provider Elo where it has a rating**")
    assert rows[1].startswith("| In-house Elo, same matches")
    assert "nan" not in report


def test_baselines_are_also_scored_on_the_matches_the_provider_rates(features):
    import pandas as pd

    from padel_predictor.training.dataset import baseline_metrics, rated_by_provider

    every_second = np.arange(len(features)) % 2 == 0
    features = features.assign(
        provider_elo_prob=pd.Series(0.7, index=features.index).where(every_second)
    )
    split = time_split(features, 0.2)
    rated = rated_by_provider(split)
    out = baseline_metrics(split)

    assert 0 < rated.sum() < len(split.y_test)
    assert out["baseline_provider_elo_rated_rows"] == rated.sum()
    expected = classification_metrics(split.y_test[rated], np.full(rated.sum(), 0.7))
    assert out["baseline_provider_elo_rated_log_loss"] == pytest.approx(expected["log_loss"])
    # The same matches for the in-house Elo, so the two rows compare like for like.
    elo_prob = 1.0 / (1.0 + 10.0 ** (-split.x_test["elo_diff"].to_numpy() / 400.0))
    assert out["baseline_elo_rated_log_loss"] == pytest.approx(
        classification_metrics(split.y_test[rated], elo_prob[rated])["log_loss"]
    )


def test_without_any_provider_rating_the_rated_baselines_are_missing(features):
    from padel_predictor.training.dataset import baseline_metrics

    out = baseline_metrics(time_split(features, 0.2))
    assert out["baseline_provider_elo_rated_rows"] == 0
    assert np.isnan(out["baseline_provider_elo_rated_log_loss"])
    assert np.isnan(out["baseline_elo_rated_log_loss"])


def test_local_registry_versions_models_and_moves_the_champion_alias(settings):
    from sklearn.dummy import DummyClassifier

    x, y = np.zeros((4, 1)), np.array([0, 1, 0, 1])
    assert registry.load_champion(settings) is None

    with registry.start_run(settings) as run:
        first = run.log_model(DummyClassifier().fit(x, y), promote=True, metadata={})
    with registry.start_run(settings) as run:
        second = run.log_model(DummyClassifier().fit(x, y), promote=False, metadata={})
    assert (first.version, second.version) == ("v1", "v2")
    assert registry.load_champion(settings).version == "v1"

    with registry.start_run(settings) as run:
        third = run.log_model(DummyClassifier().fit(x, y), promote=True, metadata={})
    champion = registry.load_champion(settings)
    assert champion.version == third.version == "v3"
    assert champion.run_id == third.run_id


def test_training_registers_and_serves_the_champion(settings, config, many_matches):
    from padel_predictor.common import storage
    from padel_predictor.inference.predict import Predictor

    features = build_feature_frame(many_matches, config.features)
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)

    result = pipeline.run(settings=settings, config=config, register=True)
    assert result.promoted
    assert Predictor.load(settings, config).model_version == result.model_version


def test_unknown_tracking_backend_is_rejected(monkeypatch):
    from padel_predictor.common.config import Settings

    monkeypatch.setenv("PADEL_TRACKING", "mlflow")
    with pytest.raises(ValueError, match="PADEL_TRACKING"):
        registry.load_champion(Settings())


def test_the_covered_comparison_ignores_matches_without_a_provider_rating():
    """Where the provider has no rating, its 0.5 must not count against it."""
    from padel_predictor.training.dataset import covered_comparison

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    # Both predictors are equally good where the provider has a rating ...
    calibrated = np.where(y == 1, 0.7, 0.3)
    provider = calibrated.copy()
    # ... and the model is near-perfect on the 200 matches the provider lacks.
    model = calibrated.copy()
    model[200:] = np.where(y[200:] == 1, 0.99, 0.01)
    provider[200:] = np.nan

    out = covered_comparison(y, model, provider, samples=300)

    assert out["covered_rows"] == 200
    assert out["covered_model_log_loss"] == pytest.approx(out["covered_provider_elo_log_loss"])
    assert out["covered_vs_provider_elo_ci_low"] <= 0 <= out["covered_vs_provider_elo_ci_high"]
    assert out["covered_beats_provider_elo"] == 0.0


def test_the_covered_comparison_is_empty_without_any_provider_rating():
    from padel_predictor.training.dataset import covered_comparison

    out = covered_comparison(np.array([0, 1]), np.array([0.4, 0.6]), np.full(2, np.nan))
    assert out["covered_rows"] == 0
    assert out["covered_beats_provider_elo"] == 0.0


# --------------------------------------------------------------------------- #
# The champion is scored only on test matches it was not trained on
# --------------------------------------------------------------------------- #
class Memoriser:
    """A champion that remembers the outcome of every match it was trained on.

    It is useless on new matches (0.5) but near-perfect on the ones it saw, so
    scoring it on its own training matches makes it look better than any
    honest model -- exactly what a backfill that moves the test window does.
    """

    def __init__(self, x, y) -> None:
        self.known = {row: int(label) for row, label in zip(self._rows(x), y, strict=True)}

    @staticmethod
    def _rows(x):
        # A missing value never equals itself, so it is keyed as None.
        return [tuple(row) for row in x.astype(object).where(x.notna(), None).to_numpy()]

    def predict_proba(self, x):
        p = np.array([{1: 0.999, 0: 0.001}.get(self.known.get(row), 0.5) for row in self._rows(x)])
        return np.column_stack([1 - p, p])


def _register_champion(settings, model, train_end, *, in_metadata: bool = True):
    with registry.start_run(settings) as run:
        if not in_metadata:
            # How models logged before train_end was artifact metadata record it.
            run.log_params({"train_end": str(train_end)})
        metadata = {"train_end": str(train_end)} if in_metadata else {}
        return run.log_model(model, promote=True, metadata=metadata)


@pytest.fixture
def small_window_config(config):
    """The synthetic test set has 180 matches; allow comparisons on a part of it."""
    config = config.model_copy(deep=True)
    config.training.promotion.min_comparison_rows = 20
    return config


def test_unseen_rows_start_after_the_champions_training_end(features):
    from padel_predictor.training.dataset import unseen_by

    split = time_split(features, 0.2)
    meta = split.meta_test
    end = meta["played_at"].iloc[len(meta) // 2]

    unseen = unseen_by(split, end)

    assert 0 < unseen.sum() < len(meta)
    assert (meta.loc[unseen, "played_at"] > end).all()
    # A tournament the champion saw part of is left out entirely, as in the split.
    seen_tournaments = set(meta.loc[meta["played_at"] <= end, "tournament"])
    assert not seen_tournaments & set(meta.loc[unseen, "tournament"])


def test_every_test_row_is_unseen_by_a_model_trained_before_the_split(features):
    from padel_predictor.training.dataset import unseen_by

    split = time_split(features, 0.2)
    assert unseen_by(split, split.train_end).all()
    assert unseen_by(split, None).all()


def test_a_champion_is_not_scored_on_matches_it_was_trained_on(
    settings, small_window_config, features
):
    from padel_predictor.common import storage
    from padel_predictor.common.features import FEATURE_COLUMNS, labelled
    from padel_predictor.training.dataset import unseen_by

    config = small_window_config
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    split = time_split(features, config.training.test_fraction)
    end = split.meta_test["played_at"].iloc[len(split.meta_test) // 2]

    # The champion was trained up to the middle of today's test window.
    seen = labelled(features).loc[lambda f: f["played_at"] <= end]
    memoriser = Memoriser(seen[list(FEATURE_COLUMNS)], seen["team_a_won"].astype(int))
    _register_champion(settings, memoriser, end)

    result = pipeline.run(settings=settings, config=config, register=True)

    # On the whole test set its memory would have kept it in place ...
    on_everything = classification_metrics(
        split.y_test, memoriser.predict_proba(split.x_test)[:, 1]
    )
    assert on_everything["log_loss"] < result.metrics["log_loss"]
    # ... scored only on matches it never saw, the trained candidate replaces it.
    assert result.comparison_rows == int(unseen_by(split, end).sum())
    assert result.promoted
    assert "after champion v1's training end" in result.reason


def test_nothing_is_promoted_without_test_matches_the_champion_has_not_seen(
    settings, small_window_config, features
):
    from padel_predictor.common import storage
    from padel_predictor.common.features import FEATURE_COLUMNS, labelled

    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    every = labelled(features)
    memoriser = Memoriser(every[list(FEATURE_COLUMNS)], every["team_a_won"].astype(int))
    _register_champion(settings, memoriser, every["played_at"].max())

    result = pipeline.run(settings=settings, config=small_window_config, register=True)

    assert result.comparison_rows == 0
    assert not result.promoted
    assert "no fair comparison" in result.reason
    assert result.model_version == "v1"


def test_a_trained_model_records_its_training_end(settings, config, features):
    from padel_predictor.common import storage

    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    pipeline.run(settings=settings, config=config, register=True)

    champion = registry.load_champion(settings)
    assert champion.train_end == time_split(features, config.training.test_fraction).train_end


def test_an_older_champion_falls_back_to_its_runs_training_end(settings):
    from sklearn.dummy import DummyClassifier

    model = DummyClassifier().fit(np.zeros((2, 1)), [0, 1])
    _register_champion(settings, model, "2026-06-28 00:00:00+00:00", in_metadata=False)

    champion = registry.load_champion(settings)
    assert champion.train_end == registry.parse_train_end("2026-06-28T00:00:00Z")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-06-28 00:00:00+00:00", "2026-06-28T00:00:00Z"),
        ("2026-06-28", "2026-06-28T00:00:00Z"),
        ({"value": "2026-06-28T10:00:00+02:00"}, "2026-06-28T08:00:00Z"),
        (None, None),
        ("", None),
        ("not a date", None),
    ],
)
def test_parse_train_end_reads_every_recorded_form(value, expected):
    import pandas as pd

    parsed = registry.parse_train_end(value)
    assert parsed == (pd.Timestamp(expected) if expected else None)


# --------------------------------------------------------------------------- #
# The provider's Elo is the first champion
# --------------------------------------------------------------------------- #
def test_the_provider_elo_model_reads_only_its_baseline_column():
    import pandas as pd

    from padel_predictor.common.features import FEATURE_COLUMNS, model_inputs
    from padel_predictor.training.baseline import ProviderEloModel

    frame = pd.DataFrame(
        {**{c: [9.0, 9.0] for c in FEATURE_COLUMNS}, "provider_elo_prob": [0.8, float("nan")]}
    )
    model = ProviderEloModel()

    assert list(model_inputs(model, frame).columns) == ["provider_elo_prob"]
    assert model.predict_proba(model_inputs(model, frame))[:, 1].tolist() == [0.8, 0.5]
    assert list(model_inputs(object(), frame).columns) == list(FEATURE_COLUMNS)


def test_an_empty_registry_starts_with_the_provider_elo_as_champion(settings, config, features):
    import joblib

    from padel_predictor.common import storage
    from padel_predictor.training.baseline import ProviderEloModel

    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    result = pipeline.run(settings=settings, config=config, register=True)

    first = Path(settings.local_registry_dir) / settings.registered_model_name / "v1"
    assert isinstance(joblib.load(first / "model.joblib"), ProviderEloModel)
    # The provider is scored on every test match, exactly as its baseline is.
    assert result.comparison_rows == result.test_rows
    assert "is the provider's Elo" in result.reason
    assert result.champion_metrics["log_loss"] == pytest.approx(
        result.baselines["baseline_provider_elo_log_loss"]
    )


def test_the_first_promotion_is_decided_where_the_provider_has_a_rating(
    settings, small_window_config, features
):
    """Beating the 0.5 an unrated match gets is not beating the provider (E1, B1)."""
    import pandas as pd

    from padel_predictor.common import storage

    config = small_window_config
    # The provider rates every fourth match, and is clearly better there.
    won = features["team_a_won"].astype("float64")
    every_fourth = pd.Series(np.arange(len(features)) % 4 == 0, index=features.index)
    features = features.assign(provider_elo_prob=(0.25 + 0.5 * won).where(every_fourth))
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)

    result = pipeline.run(settings=settings, config=config, register=True)

    rated = int(
        time_split(features, config.training.test_fraction)
        .meta_test["provider_elo_prob"]
        .notna()
        .sum()
    )
    assert config.training.promotion.min_comparison_rows <= rated < result.test_rows
    # With the unrated matches at 0.5 the model is ahead and used to be promoted ...
    assert (
        result.baselines["vs_provider_elo_log_loss_diff"]
        < -config.training.promotion.min_improvement
    )
    # ... but where the provider has a rating it is not, and that decides.
    assert result.comparison_rows == rated
    assert "the provider has a rating for" in result.reason
    assert not result.promoted
    assert registry.load_champion(settings).version == "v1"


def test_the_provider_elo_champion_is_compared_on_all_matches_without_ratings(
    settings, config, features
):
    """No ranking history (the free plan): the provider is a coin flip everywhere."""
    from padel_predictor.common import storage

    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)
    result = pipeline.run(settings=settings, config=config, register=True)

    assert result.comparison_rows == result.test_rows
    assert "which rates only 0 of them" in result.reason
    assert result.promoted


def test_a_model_that_does_not_beat_the_provider_is_not_served(settings, config, features):
    from padel_predictor.common import storage
    from padel_predictor.inference.predict import Predictor
    from padel_predictor.training.baseline import ProviderEloModel

    # A provider that is nearly always right cannot be beaten.
    won = features["team_a_won"].astype("float64")
    features = features.assign(provider_elo_prob=0.05 + 0.9 * won)
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)

    result = pipeline.run(settings=settings, config=config, register=True)

    assert not result.promoted
    predictor = Predictor.load(settings, config)
    assert isinstance(predictor.handle.model, ProviderEloModel)
    upcoming = features.head(3).assign(provider_elo_prob=[0.7, 0.2, float("nan")])
    assert predictor.probabilities(upcoming).tolist() == [0.7, 0.2, 0.5]


def test_without_the_provider_first_the_first_trained_model_is_promoted(settings, config, features):
    from padel_predictor.common import storage

    config = config.model_copy(deep=True)
    config.training.promotion.provider_elo_first_champion = False
    storage.write_snapshot(features, settings.feature_store_uri, storage.FEATURES)

    result = pipeline.run(settings=settings, config=config, register=True)

    assert result.promoted
    assert result.model_version == "v1"
    assert "no champion registered yet" in result.reason
