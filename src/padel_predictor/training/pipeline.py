"""Training pipeline: feature store -> model -> registry.

Run on a schedule and on demand by ``.github/workflows/training-pipeline.yml``.
Every run is an MLflow run: parameters, metrics, baselines and the promotion
decision are all logged, so "why is this model live?" has an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from padel_predictor.common.features import FEATURE_COLUMNS, labelled
from padel_predictor.common.logging import get_logger
from padel_predictor.training import registry
from padel_predictor.training.dataset import (
    Split,
    baseline_metrics,
    classification_metrics,
    time_split,
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

    def summary(self) -> str:
        return (
            f"train={self.train_rows} test={self.test_rows} "
            f"log_loss={self.metrics['log_loss']:.4f} "
            f"accuracy={self.metrics['accuracy']:.4f} "
            f"(elo baseline {self.baselines['baseline_elo_log_loss']:.4f}) "
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


def evaluate(model: Any, split: Split) -> dict[str, float]:
    """Score a fitted model on the holdout."""
    proba = model.predict_proba(split.x_test)[:, 1]
    return classification_metrics(split.y_test, proba)


def run(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    register: bool = True,
) -> TrainingResult:
    """Train a candidate, compare it to the champion, promote it if it is better.

    Args:
        register: when ``False``, train and evaluate but touch neither MLflow
            nor the registry. Used by the tests.
    """
    settings = settings or get_settings()
    config = config or get_pipeline_config()

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

    model = build_model(config)
    model.fit(split.x_train, split.y_train)
    metrics = evaluate(model, split)
    baselines = baseline_metrics(split)

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


def _track_and_promote(
    model: Pipeline,
    split: Split,
    metrics: dict[str, float],
    baselines: dict[str, float],
    settings: Settings,
    config: PipelineConfig,
) -> TrainingResult:
    """Log the run to MLflow and promote the candidate if it beats the champion."""
    mlflow = registry.configure(settings)
    promotion = config.training.promotion

    champion = registry.load_champion(settings)
    champion_metrics = evaluate(champion.model, split) if champion else None
    if champion_metrics:
        logger.info("champion version %s scores %s", champion.version, champion_metrics)

    promote, reason = registry.should_promote(
        candidate=metrics,
        champion=champion_metrics,
        metric=promotion.metric,
        higher_is_better=promotion.higher_is_better,
        min_improvement=promotion.min_improvement,
    )

    with mlflow.start_run(run_name="training") as run_ctx:
        mlflow.log_params(
            {
                **config.training.model.model_dump(),
                "test_fraction": config.training.test_fraction,
                "random_state": config.training.random_state,
                "features": ",".join(FEATURE_COLUMNS),
                "train_rows": len(split.x_train),
                "test_rows": len(split.x_test),
                "train_end": str(split.train_end),
                "test_start": str(split.test_start),
            }
        )
        mlflow.log_metrics({**metrics, **baselines})
        if champion_metrics:
            mlflow.log_metrics({f"champion_{k}": v for k, v in champion_metrics.items()})
        mlflow.set_tags({"promoted": str(promote), "promotion_reason": reason})

        version: str | None = champion.version if champion else None
        if promote:
            handle = registry.register_and_promote(
                model, input_example=split.x_train.head(5), settings=settings
            )
            version = handle.version
        else:
            logger.info("candidate not promoted: %s", reason)

        logger.info("mlflow run %s finished", run_ctx.info.run_id)

    result = TrainingResult(
        metrics=metrics,
        baselines=baselines,
        champion_metrics=champion_metrics,
        promoted=promote,
        reason=reason,
        model_version=version,
        train_rows=len(split.x_train),
        test_rows=len(split.x_test),
    )
    logger.info("training pipeline done | %s", result.summary())
    return result
