"""Batch inference: predict every upcoming match and write the results back.

Run on a schedule by ``.github/workflows/inference-pipeline.yml``. The on-demand
path (UI and API) shares the same :class:`~padel_predictor.inference.predict.Predictor`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.logging import get_logger
from padel_predictor.inference.predict import Predictor, load_history

logger = get_logger(__name__)


@dataclass(frozen=True)
class InferenceRunResult:
    """What one batch inference run produced."""

    predictions: int
    model_version: str
    horizon_days: int
    predictions_uri: str

    def summary(self) -> str:
        return (
            f"predictions={self.predictions} model_version={self.model_version} "
            f"horizon={self.horizon_days}d -> {self.predictions_uri}"
        )


def run(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    run_date: dt.date | None = None,
) -> InferenceRunResult:
    """Predict upcoming matches and store a dated prediction snapshot.

    Storing predictions (rather than only serving them) is what makes the system
    observable: once the matches are played, the stored probabilities can be
    joined against the actual results to track live performance and drift.
    """
    settings = settings or get_settings()
    config = config or get_pipeline_config()

    predictor = Predictor.load(settings, config)
    history = load_history(settings)
    predictions = predictor.predict_upcoming(history)

    uri = storage.write_snapshot(
        predictions, settings.feature_store_uri, storage.PREDICTIONS, run_date
    )
    result = InferenceRunResult(
        predictions=len(predictions),
        model_version=predictor.model_version,
        horizon_days=config.inference.horizon_days,
        predictions_uri=uri,
    )
    logger.info("inference pipeline done | %s", result.summary())
    return result
