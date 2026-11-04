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
    cutoff = _tournament_boundary(data, len(data) - n_test)
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


def _tournament_boundary(data: pd.DataFrame, cutoff: int) -> int:
    """Move a row cutoff to the nearest tournament boundary.

    A tournament is never split between train and test: its early rounds would
    otherwise inform the evaluation of its late rounds through Elo and form.
    Falls back to the row cutoff when no boundary leaves both sides non-empty.
    """
    if "tournament" not in data.columns or cutoff <= 0 or cutoff >= len(data):
        return cutoff
    names = data["tournament"].astype("string").fillna("").to_numpy()
    boundaries = [i for i in range(1, len(names)) if names[i] != names[i - 1]]
    if not boundaries:
        return cutoff
    best = min(boundaries, key=lambda i: (abs(i - cutoff), i))
    return best if 0 < best < len(data) else cutoff


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


def provider_elo_probabilities(split: Split) -> tuple[np.ndarray, float]:
    """The provider-Elo baseline on exactly the test matches, and its coverage.

    Uses ``provider_elo_prob`` from the feature store: the probability implied
    by the provider's Elo in the last snapshot published *before* each match
    (see :func:`padel_predictor.common.features.attach_rankings`). A test match
    without a snapshot for all four players gets 0.5 -- the provider knows
    nothing about it -- so the baseline is scored on the same rows as the model
    rather than on a friendlier subset.
    """
    if "provider_elo_prob" not in split.meta_test.columns:
        return np.full(len(split.y_test), 0.5), 0.0
    raw = pd.to_numeric(split.meta_test["provider_elo_prob"], errors="coerce").to_numpy(float)
    coverage = float(np.isfinite(raw).mean()) if len(raw) else 0.0
    return np.where(np.isfinite(raw), raw, 0.5), coverage


def baseline_metrics(split: Split) -> dict[str, float]:
    """Metrics for the baselines any real model has to beat.

    * ``majority``: always predict the more common training outcome.
    * ``elo``: the in-house Elo expectation from ``elo_diff``.
    * ``provider_elo``: the data provider's own Elo as published before each
      match -- the bar in the success criterion.
    """
    majority_rate = float(split.y_train.mean())
    majority = classification_metrics(split.y_test, np.full(len(split.y_test), majority_rate))

    elo_prob = 1.0 / (1.0 + 10.0 ** (-split.x_test["elo_diff"].to_numpy() / 400.0))
    elo = classification_metrics(split.y_test, elo_prob)

    provider_prob, coverage = provider_elo_probabilities(split)
    provider = classification_metrics(split.y_test, provider_prob)

    return {
        **{f"baseline_majority_{k}": v for k, v in majority.items()},
        **{f"baseline_elo_{k}": v for k, v in elo.items()},
        **{f"baseline_provider_elo_{k}": v for k, v in provider.items()},
        "baseline_provider_elo_coverage": coverage,
    }


def per_match_log_loss(y_true: pd.Series, proba: np.ndarray) -> np.ndarray:
    """Log loss of every single prediction, so differences can be resampled."""
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_log_loss_difference(
    y_true: pd.Series,
    model_proba: np.ndarray,
    baseline_proba: np.ndarray,
    samples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap of ``log_loss(model) - log_loss(baseline)`` on the test set.

    Matches are resampled with replacement and both predictors are scored on the
    same resample, so the interval reflects how much the *difference* moves
    with the choice of test matches. Negative means the model is better; the
    model beats the baseline only if the whole interval is below zero.
    """
    diff = per_match_log_loss(y_true, model_proba) - per_match_log_loss(y_true, baseline_proba)
    if len(diff) == 0:
        nan = float("nan")
        return {"mean": nan, "ci_low": nan, "ci_high": nan, "beats_baseline": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(samples, len(diff)))
    means = diff[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean": float(diff.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "beats_baseline": float(high < 0.0),
    }
