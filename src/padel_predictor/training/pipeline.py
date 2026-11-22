"""Training pipeline: feature store -> model -> registry.

Run on a schedule and on demand by ``.github/workflows/training-pipeline.yml``.
Every run is a tracked run (Weights & Biases in production): parameters,
metrics, baselines and the promotion decision are all logged, so "why is this
model live?" has an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import FEATURE_COLUMNS, labelled, model_inputs
from padel_predictor.common.logging import get_logger
from padel_predictor.training import registry
from padel_predictor.training.baseline import PROVIDER_ELO_RUN_NAME, ProviderEloModel
from padel_predictor.training.dataset import (
    Split,
    baseline_metrics,
    bootstrap_log_loss_difference,
    classification_metrics,
    covered_comparison,
    provider_elo_probabilities,
    rated_by_provider,
    time_split,
    unseen_by,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """What one training run produced."""

    metrics: dict[str, float]
    baselines: dict[str, float]
    champion_metrics: dict[str, float] | None
    promoted: bool
    reason: str
    model_version: str | None
    train_rows: int
    test_rows: int
    #: Test matches the champion had not been trained on, which both were scored on.
    comparison_rows: int | None = None

    @property
    def success_criterion_met(self) -> bool:
        """The model beats the provider's Elo on the matches the provider rates."""
        return bool(self.baselines["covered_beats_provider_elo"])

    def summary(self) -> str:
        b = self.baselines
        return (
            f"train={self.train_rows} test={self.test_rows} "
            f"log_loss={self.metrics['log_loss']:.4f} "
            f"accuracy={self.metrics['accuracy']:.4f} | "
            # The success criterion comes first: the matches the provider rates.
            f"rated {b['covered_rows']:.0f}: "
            f"model {b['covered_model_log_loss']:.4f} "
            f"vs provider {b['covered_provider_elo_log_loss']:.4f}, "
            f"diff {b['covered_vs_provider_elo_log_loss_diff']:+.4f} "
            f"[{b['covered_vs_provider_elo_ci_low']:+.4f}, "
            f"{b['covered_vs_provider_elo_ci_high']:+.4f}], "
            f"criterion met={self.success_criterion_met} | "
            f"all matches, unrated at 0.5: provider {b['baseline_provider_elo_log_loss']:.4f}, "
            f"diff {b['vs_provider_elo_log_loss_diff']:+.4f} "
            f"[{b['vs_provider_elo_ci_low']:+.4f}, {b['vs_provider_elo_ci_high']:+.4f}] | "
            f"promoted={self.promoted} [{self.reason}]"
        )


def build_model(config: PipelineConfig) -> Pipeline:
    """The estimator.

    Wrapped in a ``Pipeline`` so any preprocessing added later travels with the
    model into the registry, instead of living in a script the serving side
    would have to duplicate.
    """
    params = config.training.model
    return Pipeline(
        steps=[
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=params.max_iter,
                    learning_rate=params.learning_rate,
                    max_depth=params.max_depth,
                    l2_regularization=params.l2_regularization,
                    early_stopping=params.early_stopping,
                    random_state=config.training.random_state,
                ),
            )
        ]
    )


def load_training_features(settings: Settings) -> pd.DataFrame:
    """Read the newest feature snapshot the feature pipeline produced."""
    features = storage.read_latest_snapshot(settings.feature_store_uri, storage.FEATURES)
    if features.empty:
        raise RuntimeError(
            f"no feature snapshots under {settings.feature_store_uri!r}; "
            "run the feature pipeline first"
        )
    return features


def load_split(settings: Settings, config: PipelineConfig) -> Split:
    """The chronological train/test split of the newest feature snapshot."""
    features = load_training_features(settings)
    rows = len(labelled(features))
    if rows < config.training.min_training_rows:
        raise RuntimeError(
            f"only {rows} labelled matches in the feature store, "
            f"need at least {config.training.min_training_rows}; backfill more history"
        )

    split = time_split(features, config.training.test_fraction)
    logger.info(
        "training on %d matches up to %s, testing on %d from %s",
        len(split.x_train),
        split.train_end.date(),
        len(split.x_test),
        split.test_start.date(),
    )
    return split


def evaluate(model: Any, split: Split) -> dict[str, float]:
    """Score a fitted model on the holdout."""
    proba = model.predict_proba(split.x_test)[:, 1]
    return classification_metrics(split.y_test, proba)


def compare_to_provider_elo(model: Any, split: Split, config: PipelineConfig) -> dict[str, float]:
    """A bootstrap interval on the gap to the provider's Elo.

    ``covered_*`` is the success criterion: the matches the provider has a
    rating for. ``vs_provider_elo_*`` is the same gap on every test match with
    an unrated one at 0.5, kept for comparison with earlier runs.
    """
    evaluation = config.training.evaluation
    provider_prob, _ = provider_elo_probabilities(split)
    model_proba = model.predict_proba(split.x_test)[:, 1]
    result = bootstrap_log_loss_difference(
        split.y_test,
        model_proba,
        provider_prob,
        samples=evaluation.bootstrap_samples,
        confidence=evaluation.confidence,
        seed=config.training.random_state,
    )
    return {
        "vs_provider_elo_log_loss_diff": result["mean"],
        "vs_provider_elo_ci_low": result["ci_low"],
        "vs_provider_elo_ci_high": result["ci_high"],
        "beats_provider_elo": result["beats_baseline"],
        **covered_comparison(
            split.y_test,
            model_proba,
            pd.to_numeric(split.meta_test["provider_elo_prob"], errors="coerce").to_numpy(float)
            if "provider_elo_prob" in split.meta_test.columns
            else np.full(len(split.y_test), np.nan),
            samples=evaluation.bootstrap_samples,
            confidence=evaluation.confidence,
            seed=config.training.random_state,
        ),
    }


def run(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    register: bool = True,
) -> TrainingResult:
    """Train a candidate, compare it to the champion, promote it if it is better.

    Args:
        register: when ``False``, train and evaluate but touch neither the
            tracker nor the registry. Used by the tests.
    """
    settings = settings or get_settings()
    config = config or get_pipeline_config()
    split = load_split(settings, config)

    model = build_model(config)
    model.fit(split.x_train, split.y_train)
    metrics = evaluate(model, split)
    baselines = {**baseline_metrics(split), **compare_to_provider_elo(model, split, config)}

    if not register:
        return TrainingResult(
            metrics=metrics,
            baselines=baselines,
            champion_metrics=None,
            promoted=False,
            reason="registration disabled",
            model_version=None,
            train_rows=len(split.x_train),
            test_rows=len(split.x_test),
        )

    return _track_and_promote(model, split, metrics, baselines, settings, config)


@dataclass(frozen=True)
class ChampionComparison:
    """The candidate against the champion, on test matches neither was trained on."""

    promote: bool
    reason: str
    rows: int
    candidate_metrics: dict[str, float] | None = None
    champion_metrics: dict[str, float] | None = None


def compare_with_champion(
    model: Any,
    metrics: dict[str, float],
    champion: registry.ModelHandle | None,
    split: Split,
    config: PipelineConfig,
) -> ChampionComparison:
    """Decide whether the candidate replaces the champion.

    The test set is the newest fifth of the store, so a backfill of older history
    moves it back in time, and a champion trained before the backfill may have
    been fitted on matches that are now test matches. Both models are therefore
    scored only on the test matches after the champion's ``train_end``
    (:func:`~padel_predictor.training.dataset.unseen_by`); the candidate's own
    metrics in the run stay on the whole test set.

    When the champion is the provider's Elo, both are scored only on the test
    matches the provider has a rating for. On the others it predicts 0.5, and a
    model that merely beats a coin flip there would be served without having
    beaten the provider anywhere. Only when the provider rates fewer than
    ``min_comparison_rows`` test matches -- no ranking history, as on the free
    plan -- is the whole test set used, and the reason says so.

    A candidate that beats a trained champion must pass the same bar before it
    is served (:func:`beats_provider_elo_where_rated`): a registry that already
    held a trained champion never started from the provider's Elo, and beating
    that champion says nothing about beating the provider.
    """
    promotion = config.training.promotion
    rule = {
        "metric": promotion.metric,
        "higher_is_better": promotion.higher_is_better,
        "min_improvement": promotion.min_improvement,
    }
    if champion is None:
        promote, reason = registry.should_promote(candidate=metrics, champion=None, **rule)
        return ChampionComparison(promote, reason, rows=len(split.y_test))

    unseen = unseen_by(split, champion.train_end)
    rows = int(unseen.sum())
    if not getattr(champion.model, "trained_on_matches", True):
        rated = rated_by_provider(split)
        if int(rated.sum()) >= promotion.min_comparison_rows:
            unseen, rows = rated, int(rated.sum())
            scope = (
                f"the {rows} of {len(rated)} test matches the provider has a rating for "
                f"(champion {champion.version} is the provider's Elo)"
            )
        else:
            scope = (
                f"all {rows} test matches (champion {champion.version} is the provider's Elo, "
                f"which rates only {int(rated.sum())} of them)"
            )
    elif champion.train_end is None:
        logger.warning(
            "champion %s has no recorded training end; comparing on the whole test set, "
            "which may include matches it was trained on",
            champion.version,
        )
        scope = f"all {rows} test matches (champion {champion.version}'s training end unknown)"
    else:
        scope = (
            f"{rows} test matches after champion {champion.version}'s "
            f"training end {champion.train_end.date()}"
        )

    if rows < promotion.min_comparison_rows:
        return ChampionComparison(
            False,
            f"no fair comparison: only {scope}, need {promotion.min_comparison_rows}",
            rows=rows,
        )

    y = split.y_test.loc[unseen]
    # The whole test rows: the provider-Elo champion reads a baseline column.
    rows_frame = split.meta_test.loc[unseen]
    candidate_metrics = classification_metrics(
        y, model.predict_proba(model_inputs(model, rows_frame))[:, 1]
    )
    champion_metrics = classification_metrics(
        y, champion.model.predict_proba(model_inputs(champion.model, rows_frame))[:, 1]
    )
    logger.info("on %s: candidate %s, champion %s", scope, candidate_metrics, champion_metrics)
    promote, reason = registry.should_promote(
        candidate=candidate_metrics, champion=champion_metrics, **rule
    )
    reason = f"{reason} on {scope}"
    if promote and getattr(champion.model, "trained_on_matches", True):
        gate = beats_provider_elo_where_rated(model, split, config)
        if gate is not None and not gate[0]:
            promote, reason = False, f"{reason}, but {gate[1]}"
    return ChampionComparison(
        promote,
        reason,
        rows=rows,
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
    )


def beats_provider_elo_where_rated(
    model: Any, split: Split, config: PipelineConfig
) -> tuple[bool, str] | None:
    """Whether a candidate beats the provider's Elo on the rated test matches.

    The bar every trained model passes before it is served, by the same
    ``min_improvement`` as any promotion. ``None`` when the bar does not apply:
    ``provider_elo_first_champion`` is off, or the provider rates fewer than
    ``min_comparison_rows`` test matches.
    """
    promotion = config.training.promotion
    rated = rated_by_provider(split)
    rows = int(rated.sum())
    if not promotion.provider_elo_first_champion or rows < promotion.min_comparison_rows:
        return None
    y = split.y_test.loc[rated]
    frame = split.meta_test.loc[rated]
    provider = ProviderEloModel()
    promote, reason = registry.should_promote(
        candidate=classification_metrics(y, model.predict_proba(model_inputs(model, frame))[:, 1]),
        champion=classification_metrics(
            y, provider.predict_proba(model_inputs(provider, frame))[:, 1]
        ),
        metric=promotion.metric,
        higher_is_better=promotion.higher_is_better,
        min_improvement=promotion.min_improvement,
    )
    return promote, (
        f"against the provider's Elo on the {rows} test matches it has a rating for: {reason}"
    )


def _track_and_promote(
    model: Pipeline,
    split: Split,
    metrics: dict[str, float],
    baselines: dict[str, float],
    settings: Settings,
    config: PipelineConfig,
) -> TrainingResult:
    """Log the run and promote the candidate if it beats the champion."""
    champion = registry.load_champion(settings)
    if champion is None and config.training.promotion.provider_elo_first_champion:
        # A trained model is served only once it beats the provider's own Elo.
        champion = registry.register_first_champion(
            settings,
            ProviderEloModel(),
            run_name=PROVIDER_ELO_RUN_NAME,
            params={"model": "provider_elo", "data_plan": settings.api_plan},
        )
    comparison = compare_with_champion(model, metrics, champion, split, config)
    promote, reason = comparison.promote, comparison.reason
    champion_metrics = comparison.champion_metrics

    with registry.start_run(settings, name="training") as run:
        run.log_params(
            {
                **config.training.model.model_dump(),
                "test_fraction": config.training.test_fraction,
                "random_state": config.training.random_state,
                "features": ",".join(FEATURE_COLUMNS),
                "train_rows": len(split.x_train),
                "test_rows": len(split.x_test),
                "train_end": str(split.train_end),
                "test_start": str(split.test_start),
                # Free- and paid-plan runs are kept side by side in W&B.
                "data_plan": settings.api_plan,
            }
        )
        run.log_metrics({**metrics, **baselines, "comparison_rows": comparison.rows})
        if champion_metrics and comparison.candidate_metrics:
            # Both on the same test matches: those the champion was not trained on.
            run.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
            run.log_metrics(
                {f"comparison_candidate_{k}": v for k, v in comparison.candidate_metrics.items()}
            )
        run.set_tags(
            {
                "promoted": str(promote),
                "promotion_reason": reason,
                # The criterion: the gap on the matches the provider rates.
                "success_criterion_met": str(bool(baselines["covered_beats_provider_elo"])),
                "beats_provider_elo_where_covered": str(
                    bool(baselines["covered_beats_provider_elo"])
                ),
                # On all test matches, an unrated one at 0.5: flatters the model.
                "beats_provider_elo": str(bool(baselines["beats_provider_elo"])),
                "data_plan": settings.api_plan,
            }
        )

        # Every candidate is versioned; only a promoted one gets the champion alias.
        handle = run.log_model(
            model,
            promote=promote,
            # train_end lets a later run score this model only on matches it never saw.
            metadata={**metrics, "promotion_reason": reason, "train_end": str(split.train_end)},
        )
        version: str | None = (
            handle.version if promote else (champion.version if champion else None)
        )
        if not promote:
            logger.info("candidate %s not promoted: %s", handle.version, reason)
        logger.info("training run %s finished", run.run_id)

    result = TrainingResult(
        metrics=metrics,
        baselines=baselines,
        champion_metrics=champion_metrics,
        promoted=promote,
        reason=reason,
        model_version=version,
        train_rows=len(split.x_train),
        test_rows=len(split.x_test),
        comparison_rows=comparison.rows,
    )
    logger.info("training pipeline done | %s", result.summary())
    return result
