"""Experiments on the training pipeline's own split: ablations, training windows, Elo K.

All retrain the model in variants and score every variant the way a candidate
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
``elo-k``
    Rebuilds the features from the raw matches with each K-factor in
    ``training.elo_k.factors`` for the in-house Elo. Reference: the K-factor in
    ``features.elo_k_factor``. The K to use is chosen on validation folds inside
    the training period (each fold is fitted on everything before it), never on
    the test set: the test columns only show what the choice is worth. The
    report also gives the paired interval of the chosen K against the current
    one on the validation matches, so a pick within noise is named as one.

Each variant is a tracked run (the experiment is its job type and tag), so they
sit next to each other in W&B. No variant is logged as a model or promoted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
import pandas as pd

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import FEATURE_COLUMNS, build_feature_frame
from padel_predictor.common.logging import get_logger
from padel_predictor.common.schemas import deduplicate_matches, deduplicate_rankings
from padel_predictor.training import registry
from padel_predictor.training.dataset import (
    Split,
    baseline_metrics,
    bootstrap_log_loss_difference,
    classification_metrics,
    per_match_log_loss,
    time_split,
    tournament_boundary,
    with_features,
)
from padel_predictor.training.pipeline import build_model, compare_to_provider_elo, load_split

logger = get_logger(__name__)

ABLATION = "ablation"
TRAINING_WINDOW = "training-window"
ELO_K = "elo-k"


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
    elo_k: float | None = None
    #: Log loss on the validation folds inside the training period (``elo-k``).
    validation_log_loss: float | None = None
    #: Log loss of the in-house Elo alone on the test matches the provider rates.
    elo_rated_log_loss: float | None = None
    #: The variant the validation folds pick.
    selected: bool = False


@dataclass(frozen=True)
class ExperimentReport:
    experiment: str
    reference: str
    variants: list[Variant]
    test_rows: int
    test_start: str
    skipped: list[str] = field(default_factory=list)
    #: How the variant marked as chosen was picked, if the experiment picks one.
    selection: str = ""

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

        def number(value: float | None) -> str:
            return "–" if value is None or not np.isfinite(value) else f"{value:.4f}"

        # Only an experiment that picks a variant has validation folds.
        validated = any(v.validation_log_loss is not None for v in self.variants)
        lines = [
            f"**{self.experiment}**: tested on {self.test_rows} matches from "
            f"{self.test_start}. Gaps are log loss differences with 95 % bootstrap "
            "intervals; negative means the variant is better.",
            "",
            "| Variant | Trained on | "
            + ("Validation log loss | In-house Elo alone, rated matches | " if validated else "")
            + "Log loss | Accuracy | vs provider Elo, rated matches | "
            f"vs provider Elo, all (unrated at 0.5) | vs {self.reference} |",
            "| --- | --- | " + ("---: | ---: | " if validated else "") + "---: | ---: | ---: | "
            "---: | ---: |",
        ]
        for v in self.variants:
            extra = (
                f"{number(v.validation_log_loss)} | {number(v.elo_rated_log_loss)} | "
                if validated
                else ""
            )
            lines.append(
                f"| {v.label}{' (chosen)' if v.selected else ''} | "
                f"{v.train_rows} from {v.train_start} | {extra}"
                f"{v.metrics['log_loss']:.4f} | {v.metrics['accuracy']:.1%} | "
                f"{provider(v, 'covered_')} | {provider(v, '')} | "
                f"{gap(v.vs_reference, 'mean', 'ci_low', 'ci_high')} |"
            )
        if self.selection:
            lines += ["", self.selection]
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
                "elo_k": variant.elo_k or "",
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
        if variant.validation_log_loss is not None:
            metrics["validation_log_loss"] = variant.validation_log_loss
        if variant.elo_rated_log_loss is not None and np.isfinite(variant.elo_rated_log_loss):
            metrics["in_house_elo_rated_log_loss"] = variant.elo_rated_log_loss
        run.log_metrics(metrics)
        tags = {"experiment": experiment, "variant": variant.label, "data_plan": settings.api_plan}
        if variant.validation_log_loss is not None:
            tags["chosen_on_validation"] = str(variant.selected)
        run.set_tags(tags)


def _report(
    experiment: str,
    reference: str,
    variants: list[Variant],
    split: Split,
    settings: Settings,
    track: bool,
    skipped: list[str] | None = None,
    selection: str = "",
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
        selection=selection,
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
    # Hopsworks returns timestamps without a time zone; they are UTC.
    train_end = pd.Timestamp(split.train_end)
    train_end = (
        train_end.tz_localize("UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")
    )
    everything = np.ones(len(split.y_train), dtype=bool)
    reference, proba = _fit_and_score(
        split, "all history", FEATURE_COLUMNS, everything, config, None
    )

    variants, skipped = [reference], []
    for m in months:
        mask = (played >= train_end - pd.DateOffset(months=m)).to_numpy()
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


def validation_predictions(
    split: Split, config: PipelineConfig, folds: int
) -> tuple[pd.Series, np.ndarray]:
    """Outcomes and predictions on forward-chaining folds inside the training period.

    The training matches are cut into ``folds + 1`` chronological blocks, each
    cut moved to a tournament boundary as in the train/test split. Every block
    but the first is predicted by a model fitted on all blocks before it, and
    the blocks are pooled. The test set is not touched.
    """
    if folds < 1:
        raise ValueError(f"validation_folds must be at least 1, got {folds}")
    rows = len(split.y_train)
    edges = [
        tournament_boundary(split.meta_train, round(rows * i / (folds + 1)))
        for i in range(1, folds + 1)
    ]
    outcomes, predictions = [], []
    for start, end in pairwise([*edges, rows]):
        if start <= 0 or end <= start:
            continue
        model = build_model(config).fit(split.x_train.iloc[:start], split.y_train.iloc[:start])
        outcomes.append(split.y_train.iloc[start:end])
        predictions.append(model.predict_proba(split.x_train.iloc[start:end])[:, 1])
    if not outcomes:
        raise ValueError(f"{rows} training matches are too few for {folds} validation folds")
    return pd.concat(outcomes, ignore_index=True), np.concatenate(predictions)


def run_elo_k(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    track: bool = True,
) -> ExperimentReport:
    """The model on features built with each K-factor for the in-house Elo.

    The features are rebuilt from the raw matches and rankings in the store by
    :func:`~padel_predictor.common.features.build_feature_frame`, exactly as the
    feature pipeline builds them, with only ``elo_k_factor`` changed. The split
    is the same matches for every K, so the variants are paired.
    """
    settings = settings or get_settings()
    config = config or get_pipeline_config()
    current = float(config.features.elo_k_factor)
    factors = sorted({float(k) for k in config.training.elo_k.factors} | {current})
    if any(k <= 0 for k in factors):
        raise ValueError(f"elo_k.factors must be positive, got {factors}")
    folds = config.training.elo_k.validation_folds

    raw = storage.read_dataset(settings.feature_store_uri, storage.RAW_MATCHES)
    if raw.empty:
        raise RuntimeError(
            f"no raw matches under {settings.feature_store_uri!r}; run the feature pipeline first"
        )
    history = deduplicate_matches(raw)
    rankings = deduplicate_rankings(
        storage.read_dataset(settings.feature_store_uri, storage.RAW_RANKINGS)
    )

    splits: dict[float, Split] = {}
    for k in factors:
        feature_config = config.features.model_copy(update={"elo_k_factor": k})
        features = build_feature_frame(history, feature_config, rankings=rankings)
        splits[k] = time_split(features, config.training.test_fraction)
    reference_split = splits[current]
    for k, split in splits.items():
        # K changes feature values only; the matches on each side must not move.
        if not split.meta_test["match_id"].equals(reference_split.meta_test["match_id"]):
            raise RuntimeError(f"K = {k:g} changed which matches are in the test set")

    folded = {k: validation_predictions(split, config, folds) for k, split in splits.items()}
    validation = {k: float(per_match_log_loss(y, p).mean()) for k, (y, p) in folded.items()}
    chosen = min(factors, key=lambda k: (validation[k], abs(k - current)))

    everything = np.ones(len(reference_split.y_train), dtype=bool)

    def score(k: float, reference: np.ndarray | None) -> tuple[Variant, np.ndarray]:
        return _fit_and_score(
            splits[k],
            f"K = {k:g}",
            FEATURE_COLUMNS,
            everything,
            config,
            reference,
            elo_k=k,
            validation_log_loss=validation[k],
            elo_rated_log_loss=baseline_metrics(splits[k])["baseline_elo_rated_log_loss"],
            selected=k == chosen,
        )

    reference, proba = score(current, None)
    variants = [reference] + [score(k, proba)[0] for k in factors if k != current]
    selection = (
        f"Lowest log loss on {folds} validation folds inside the training period "
        f"({len(folded[current][0])} matches): K = {chosen:g}"
    )
    if chosen == current:
        selection += ", the current value."
    else:
        # The same paired interval as on the test set, so a lucky pick shows.
        evaluation = config.training.evaluation
        gap = bootstrap_log_loss_difference(
            folded[chosen][0],
            folded[chosen][1],
            folded[current][1],
            samples=evaluation.bootstrap_samples,
            confidence=evaluation.confidence,
            seed=config.training.random_state,
        )
        selection += (
            f", against K = {current:g} now: {gap['mean']:+.4f} "
            f"[{gap['ci_low']:+.4f}, {gap['ci_high']:+.4f}] on those matches, "
            + (
                "reliably better."
                if gap["beats_baseline"]
                else "not distinguishable from the current value."
            )
        )
    selection += " The test columns did not enter the choice."
    return _report(
        ELO_K, f"K = {current:g}", variants, reference_split, settings, track, selection=selection
    )
