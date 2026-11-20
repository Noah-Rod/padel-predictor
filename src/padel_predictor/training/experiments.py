"""Experiments on the training pipeline's own split: ablations and training windows.

Both retrain the model in variants and score every variant the way a candidate
is scored: against the provider's Elo on all test matches and on the rated ones
alone, each with the paired bootstrap interval of the success criterion. Each
variant is also compared with a reference model on the same matches; a gap
whose interval lies above zero means the variant is worse than the reference.

``ablation``
    Drops each feature in ``training.ablation.drop`` (partnership, form, rest)
    in turn. Reference: the model on all features.
``training-window``
    Fits the model only on the matches of the last N months before the test set,
    for each N in ``training_window.months``. Reference: the model on all
    history. The features themselves are still built from the whole history --
    Elo and form are replayed from the first match -- so this measures only
    whether *fitting* on old matches helps or hurts.

Each variant is a tracked run (the experiment is its job type and tag), so they
sit next to each other in W&B. No variant is logged as a model or promoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import FEATURE_COLUMNS
from padel_predictor.common.logging import get_logger
from padel_predictor.training import registry
from padel_predictor.training.dataset import (
    Split,
    bootstrap_log_loss_difference,
    classification_metrics,
    with_features,
)
from padel_predictor.training.pipeline import build_model, compare_to_provider_elo, load_split

logger = get_logger(__name__)

ABLATION = "ablation"
TRAINING_WINDOW = "training-window"


@dataclass(frozen=True)
class Variant:
    """One retrained model and how it scores on the test set."""

    label: str
    features: tuple[str, ...]
    train_rows: int
    train_start: str
    metrics: dict[str, float]
    vs_provider: dict[str, float]
    #: Log loss of this variant minus the reference model's, on the same matches.
    vs_reference: dict[str, float] | None
    dropped: str | None = None
    months: int | None = None


@dataclass(frozen=True)
class ExperimentReport:
    experiment: str
    reference: str
    variants: list[Variant]
    test_rows: int
    test_start: str
    skipped: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        def gap(result: dict[str, float] | None, mean: str, low: str, high: str) -> str:
            # Missing: the reference row itself, or no rated match in the test set.
            if result is None or not np.isfinite(result[mean]):
                return "–"
            return f"{result[mean]:+.4f} [{result[low]:+.4f}, {result[high]:+.4f}]"

        def provider(v: Variant, prefix: str) -> str:
            return gap(
                v.vs_provider,
                f"{prefix}vs_provider_elo_log_loss_diff",
                f"{prefix}vs_provider_elo_ci_low",
                f"{prefix}vs_provider_elo_ci_high",
            )

        lines = [
            f"**{self.experiment}**: tested on {self.test_rows} matches from "
            f"{self.test_start}. Gaps are log loss differences with 95 % bootstrap "
            "intervals; negative means the variant is better.",
            "",
            "| Variant | Trained on | Log loss | Accuracy | vs provider Elo, rated matches | "
            f"vs provider Elo, all (unrated at 0.5) | vs {self.reference} |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for v in self.variants:
            lines.append(
                f"| {v.label} | {v.train_rows} from {v.train_start} | "
                f"{v.metrics['log_loss']:.4f} | {v.metrics['accuracy']:.1%} | "
                f"{provider(v, 'covered_')} | {provider(v, '')} | "
                f"{gap(v.vs_reference, 'mean', 'ci_low', 'ci_high')} |"
            )
        if self.skipped:
            lines += ["", "Skipped: " + "; ".join(self.skipped) + "."]
        return "\n".join(lines)


def _fit_and_score(
    split: Split,
    label: str,
    columns: tuple[str, ...],
    train_mask: np.ndarray,
    config: PipelineConfig,
    reference: np.ndarray | None,
    **extra: object,
) -> tuple[Variant, np.ndarray]:
    sub = with_features(split, columns)
    x_train, y_train = sub.x_train.loc[train_mask], sub.y_train.loc[train_mask]
    model = build_model(config).fit(x_train, y_train)
    proba = model.predict_proba(sub.x_test)[:, 1]

    vs_reference = None
    if reference is not None:
        evaluation = config.training.evaluation
        vs_reference = bootstrap_log_loss_difference(
            sub.y_test,
            proba,
            reference,
            samples=evaluation.bootstrap_samples,
            confidence=evaluation.confidence,
            seed=config.training.random_state,
        )
    played = split.meta_train.loc[train_mask, "played_at"] if len(split.meta_train) else None
    variant = Variant(
        label=label,
        features=columns,
        train_rows=int(train_mask.sum()),
        train_start=str(played.min().date()) if played is not None and len(played) else "–",
        metrics=classification_metrics(sub.y_test, proba),
        vs_provider=compare_to_provider_elo(model, sub, config),
        vs_reference=vs_reference,
        **extra,
    )
    logger.info("%s %s: log_loss %.4f", label, variant.train_rows, variant.metrics["log_loss"])
    return variant, proba


def _track(experiment: str, variant: Variant, split: Split, settings: Settings) -> None:
    slug = re.sub(r"[^a-z0-9]+", "-", variant.label.lower()).strip("-")
    with registry.start_run(settings, name=f"{experiment}-{slug}", job_type=experiment) as run:
        run.log_params(
            {
                "variant": variant.label,
                "features": ",".join(variant.features),
                "dropped": variant.dropped or "",
                "train_months": variant.months or "",
                "train_rows": variant.train_rows,
                "train_start": variant.train_start,
                "test_rows": len(split.x_test),
                "test_start": str(split.test_start),
                "data_plan": settings.api_plan,
            }
        )
        metrics = {**variant.metrics, **variant.vs_provider}
        if variant.vs_reference:
            metrics.update(
                {
                    "vs_reference_log_loss_diff": variant.vs_reference["mean"],
                    "vs_reference_ci_low": variant.vs_reference["ci_low"],
                    "vs_reference_ci_high": variant.vs_reference["ci_high"],
                }
            )
        run.log_metrics(metrics)
        run.set_tags(
            {"experiment": experiment, "variant": variant.label, "data_plan": settings.api_plan}
        )


def _report(
    experiment: str,
    reference: str,
    variants: list[Variant],
    split: Split,
    settings: Settings,
    track: bool,
    skipped: list[str] | None = None,
) -> ExperimentReport:
    if track:
        for variant in variants:
            _track(experiment, variant, split, settings)
    return ExperimentReport(
        experiment=experiment,
        reference=reference,
        variants=variants,
        test_rows=len(split.x_test),
        test_start=str(split.test_start.date()),
        skipped=skipped or [],
    )


def run_ablation(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    track: bool = True,
) -> ExperimentReport:
    """The model on all features, then one variant per dropped feature."""
    settings = settings or get_settings()
    config = config or get_pipeline_config()
    drop = list(config.training.ablation.drop)
    unknown = [f for f in drop if f not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"cannot drop {unknown}: not in {FEATURE_COLUMNS}")

    split = load_split(settings, config)
    everything = np.ones(len(split.y_train), dtype=bool)
    reference, proba = _fit_and_score(
        split, "none (all features)", FEATURE_COLUMNS, everything, config, None
    )
    variants = [reference]
    for feature in drop:
        columns = tuple(c for c in FEATURE_COLUMNS if c != feature)
        variant, _ = _fit_and_score(
            split, feature, columns, everything, config, proba, dropped=feature
        )
        variants.append(variant)
    return _report(ABLATION, "all features", variants, split, settings, track)


def run_training_window(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    track: bool = True,
) -> ExperimentReport:
    """The model on all history, then one variant per training window in months."""
    settings = settings or get_settings()
    config = config or get_pipeline_config()
    months = sorted(set(config.training.training_window.months), reverse=True)
    if any(m <= 0 for m in months):
        raise ValueError(f"training_window.months must be positive, got {months}")

    split = load_split(settings, config)
    played = pd.to_datetime(split.meta_train["played_at"], utc=True)
    everything = np.ones(len(split.y_train), dtype=bool)
    reference, proba = _fit_and_score(
        split, "all history", FEATURE_COLUMNS, everything, config, None
    )

    variants, skipped = [reference], []
    for m in months:
        mask = (played >= split.train_end - pd.DateOffset(months=m)).to_numpy()
        if mask.all():
            skipped.append(f"{m} months (covers all history)")
            continue
        if mask.sum() < config.training.min_training_rows:
            skipped.append(f"{m} months (only {int(mask.sum())} matches)")
            continue
        variant, _ = _fit_and_score(
            split, f"last {m} months", FEATURE_COLUMNS, mask, config, proba, months=m
        )
        variants.append(variant)
    return _report(TRAINING_WINDOW, "all history", variants, split, settings, track, skipped)
