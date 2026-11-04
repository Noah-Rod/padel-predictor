"""What is in the feature store: counts per season and tier, and the baseline bar.

``padel data-report`` prints this as Markdown, so the numbers in a milestone
summary come from the store itself rather than from an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import labelled
from padel_predictor.common.schemas import deduplicate_matches
from padel_predictor.features.pipeline import load_rankings


@dataclass(frozen=True)
class DataReport:
    counts: pd.DataFrame
    ranking_rows: int
    ranking_snapshots: int
    first_match: str
    last_match: str
    baseline: dict[str, float]
    test_rows: int
    test_start: str

    def to_markdown(self) -> str:
        lines = ["| Season | Tier | Labelled | Upcoming |", "| --- | --- | ---: | ---: |"]
        for row in self.counts.itertuples(index=False):
            lines.append(f"| {row.season} | {row.tier} | {row.labelled} | {row.upcoming} |")
        total_l, total_u = int(self.counts["labelled"].sum()), int(self.counts["upcoming"].sum())
        lines.append(f"| **all** | | **{total_l}** | **{total_u}** |")
        lines += [
            "",
            f"Matches from {self.first_match} to {self.last_match}; "
            f"{self.ranking_snapshots} weekly ranking snapshots ({self.ranking_rows} player rows).",
        ]
        if self.baseline:
            b = self.baseline
            lines += [
                "",
                f"Out-of-time test set: {self.test_rows} matches from {self.test_start}.",
                "",
                "| Baseline on the test set | Log loss | Accuracy |",
                "| --- | ---: | ---: |",
                f"| Provider Elo (as published before each match, "
                f"coverage {b['baseline_provider_elo_coverage']:.0%}) "
                f"| {b['baseline_provider_elo_log_loss']:.4f} "
                f"| {b['baseline_provider_elo_accuracy']:.3f} |",
                f"| In-house Elo | {b['baseline_elo_log_loss']:.4f} "
                f"| {b['baseline_elo_accuracy']:.3f} |",
                f"| Majority class | {b['baseline_majority_log_loss']:.4f} "
                f"| {b['baseline_majority_accuracy']:.3f} |",
            ]
        return "\n".join(lines)


def build_report(
    settings: Settings | None = None, config: PipelineConfig | None = None
) -> DataReport:
    """Summarise the raw history and score the baselines on the real test split."""
    from padel_predictor.training.dataset import baseline_metrics, time_split

    settings = settings or get_settings()
    config = config or get_pipeline_config()
    raw = storage.read_dataset(settings.feature_store_uri, storage.RAW_MATCHES)
    if raw.empty:
        raise RuntimeError("the feature store is empty; run a backfill first")
    history = deduplicate_matches(raw)

    frame = history.assign(
        season=history["played_at"].dt.year,
        labelled=history["winner"].notna().astype(int),
        upcoming=history["winner"].isna().astype(int),
    )
    counts = (
        frame.groupby(["season", "tier"], dropna=False)[["labelled", "upcoming"]]
        .sum()
        .reset_index()
        .sort_values(["season", "tier"])
    )

    rankings = load_rankings(settings.feature_store_uri)
    features = storage.read_latest_snapshot(settings.feature_store_uri, storage.FEATURES)

    baseline: dict[str, float] = {}
    test_rows, test_start = 0, "-"
    if not features.empty and len(labelled(features)) >= 2:
        split = time_split(features, config.training.test_fraction)
        baseline = baseline_metrics(split)
        test_rows, test_start = len(split.x_test), str(split.test_start.date())

    return DataReport(
        counts=counts,
        ranking_rows=len(rankings),
        ranking_snapshots=int(rankings["snapshot_date"].nunique()) if not rankings.empty else 0,
        first_match=str(history["played_at"].min().date()),
        last_match=str(history["played_at"].max().date()),
        baseline=baseline,
        test_rows=test_rows,
        test_start=test_start,
    )
