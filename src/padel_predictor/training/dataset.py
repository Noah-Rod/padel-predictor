"""Turning the feature store into a train/test split, and scoring it.

The split is strictly chronological. A random split would let a match from
March inform a prediction about a match in February, which inflates every metric
and is the single easiest way to fool yourself in this kind of project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from padel_predictor.common.features import FEATURE_COLUMNS, TARGET_COLUMN, labelled
from padel_predictor.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Split:
    """A chronological train/test split of the labelled feature rows."""

    x_train: pd.DataFrame
    y_train: pd.Series
    x_test: pd.DataFrame
    y_test: pd.Series
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    meta_test: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.x_train), "test": len(self.x_test)}


def time_split(features: pd.DataFrame, test_fraction: float = 0.2) -> Split:
    """Split labelled features by time: oldest rows train, newest rows test."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    data = labelled(features).sort_values(["played_at", "match_id"], kind="stable")
    if data.empty:
        raise ValueError("no labelled feature rows available to train on")

    n_test = max(1, int(round(len(data) * test_fraction)))
    if n_test >= len(data):
        raise ValueError(
            f"test_fraction {test_fraction} leaves no training rows for {len(data)} matches"
        )
    cutoff = len(data) - n_test
    train, test = data.iloc[:cutoff], data.iloc[cutoff:]

    return Split(
        x_train=train[list(FEATURE_COLUMNS)].reset_index(drop=True),
        y_train=train[TARGET_COLUMN].astype(int).reset_index(drop=True),
        x_test=test[list(FEATURE_COLUMNS)].reset_index(drop=True),
        y_test=test[TARGET_COLUMN].astype(int).reset_index(drop=True),
        train_end=train["played_at"].max(),
        test_start=test["played_at"].min(),
        test_end=test["played_at"].max(),
        meta_test=test.reset_index(drop=True),
    )


def classification_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Metrics for a binary probabilistic classifier.

    ``roc_auc`` is undefined when the holdout contains a single class, in which
    case it is reported as ``nan`` rather than crashing the run.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": auc,
        "brier": float(brier_score_loss(y, p)),
    }


def baseline_metrics(split: Split) -> dict[str, float]:
    """Metrics for the baselines any real model has to beat.

    * ``majority``: always predict the more common training outcome.
    * ``elo``: predict the team with the higher Elo rating, using the Elo
      expectation itself as the probability. This is the honest baseline for
      match prediction -- a model that cannot beat it has learned nothing.
    """
    majority_rate = float(split.y_train.mean())
    majority = classification_metrics(split.y_test, np.full(len(split.y_test), majority_rate))

    elo_prob = 1.0 / (1.0 + 10.0 ** (-split.x_test["elo_diff"].to_numpy() / 400.0))
    elo = classification_metrics(split.y_test, elo_prob)

    return {
        **{f"baseline_majority_{k}": v for k, v in majority.items()},
        **{f"baseline_elo_{k}": v for k, v in elo.items()},
    }
