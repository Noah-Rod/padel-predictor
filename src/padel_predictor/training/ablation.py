"""Ablations: what each feature adds to the model.

The model is retrained without each feature in ``training.ablation.drop``
(partnership, form, rest), one at a time, on exactly the split the training
pipeline uses. Every variant is scored the way a candidate is: against the
provider's Elo on all test matches and on the rated ones alone, each with the
paired bootstrap interval of the success criterion. It is also compared with
the model on all features, on the same matches: a gap whose interval lies
above zero means dropping the feature makes the model worse.

Each variant is a tracked run (job type ``ablation``, tagged ``ablation``), so
they sit next to each other in W&B. No variant is logged as a model or
promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

ALL_FEATURES = "none (all features)"


@dataclass(frozen=True)
class AblationResult:
    """One variant: the model without ``dropped`` (``None``: on every feature)."""

    dropped: str | None
    features: tuple[str, ...]
    metrics: dict[str, float]
    vs_provider: dict[str, float]
    #: log loss of this variant minus the all-feature model's, on the same matches.
    vs_all_features: dict[str, float] | None

    @property
    def label(self) -> str:
        return self.dropped or ALL_FEATURES


@dataclass(frozen=True)
class AblationReport:
    results: list[AblationResult]
    train_rows: int
    test_rows: int
    test_start: str

    def to_markdown(self) -> str:
        def gap(mean: float, low: float, high: float) -> str:
            if not np.isfinite(mean):  # e.g. no rated match in the test set
                return "–"
            return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"

        lines = [
            f"Ablations: trained on {self.train_rows} matches, "
            f"tested on {self.test_rows} from {self.test_start}. "
            "Gaps are log loss differences with 95 % bootstrap intervals; "
            "negative means the variant is better.",
            "",
            "| Dropped | Log loss | Accuracy | vs provider Elo, all | "
            "vs provider Elo, rated only | vs all features |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for r in self.results:
            p = r.vs_provider
            full = (
                gap(
                    r.vs_all_features["mean"],
                    r.vs_all_features["ci_low"],
                    r.vs_all_features["ci_high"],
                )
                if r.vs_all_features
                else "–"
            )
            lines.append(
                f"| {r.label} | {r.metrics['log_loss']:.4f} | {r.metrics['accuracy']:.1%} | "
                + gap(
                    p["vs_provider_elo_log_loss_diff"],
                    p["vs_provider_elo_ci_low"],
                    p["vs_provider_elo_ci_high"],
                )
                + " | "
                + gap(
                    p["covered_vs_provider_elo_log_loss_diff"],
                    p["covered_vs_provider_elo_ci_low"],
                    p["covered_vs_provider_elo_ci_high"],
                )
                + f" | {full} |"
            )
        return "\n".join(lines)


def _variant(
    split: Split,
    dropped: str | None,
    config: PipelineConfig,
    reference: np.ndarray | None,
) -> tuple[AblationResult, np.ndarray]:
    columns = tuple(c for c in FEATURE_COLUMNS if c != dropped)
    sub = with_features(split, columns)
    model = build_model(config).fit(sub.x_train, sub.y_train)
    proba = model.predict_proba(sub.x_test)[:, 1]

    vs_all = None
    if reference is not None:
        evaluation = config.training.evaluation
        vs_all = bootstrap_log_loss_difference(
            sub.y_test,
            proba,
            reference,
            samples=evaluation.bootstrap_samples,
            confidence=evaluation.confidence,
            seed=config.training.random_state,
        )
    result = AblationResult(
        dropped=dropped,
        features=columns,
        metrics=classification_metrics(sub.y_test, proba),
        vs_provider=compare_to_provider_elo(model, sub, config),
        vs_all_features=vs_all,
    )
    return result, proba


def _track(result: AblationResult, split: Split, settings: Settings) -> None:
    name = f"ablation-no-{result.dropped}" if result.dropped else "ablation-all-features"
    with registry.start_run(settings, name=name, job_type="ablation") as run:
        run.log_params(
            {
                "dropped": result.label,
                "features": ",".join(result.features),
                "train_rows": len(split.x_train),
                "test_rows": len(split.x_test),
                "test_start": str(split.test_start),
                "data_plan": settings.api_plan,
            }
        )
        metrics = {**result.metrics, **result.vs_provider}
        if result.vs_all_features:
            metrics.update(
                {
                    "vs_all_features_log_loss_diff": result.vs_all_features["mean"],
                    "vs_all_features_ci_low": result.vs_all_features["ci_low"],
                    "vs_all_features_ci_high": result.vs_all_features["ci_high"],
                }
            )
        run.log_metrics(metrics)
        run.set_tags({"ablation": "True", "dropped": result.label, "data_plan": settings.api_plan})


def run(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    track: bool = True,
) -> AblationReport:
    """Train the all-feature model and one variant per dropped feature."""
    settings = settings or get_settings()
    config = config or get_pipeline_config()
    drop = list(config.training.ablation.drop)
    unknown = [f for f in drop if f not in FEATURE_COLUMNS]
    if unknown:
        raise ValueError(f"cannot drop {unknown}: not in {FEATURE_COLUMNS}")

    split = load_split(settings, config)
    baseline, reference = _variant(split, None, config, reference=None)
    results = [baseline] + [_variant(split, f, config, reference)[0] for f in drop]

    if track:
        for result in results:
            _track(result, split, settings)
    for result in results:
        logger.info("ablation %s: log_loss %.4f", result.label, result.metrics["log_loss"])

    return AblationReport(
        results=results,
        train_rows=len(split.x_train),
        test_rows=len(split.x_test),
        test_start=str(split.test_start.date()),
    )
