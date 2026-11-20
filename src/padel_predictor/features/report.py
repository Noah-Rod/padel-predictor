"""What is in the feature store: counts per season and tier, and the baseline bar.

``padel data-report`` prints this as Markdown, so the numbers in a milestone
summary come from the store itself rather than from an estimate. It also lists
when each weekly ranking was first seen by a live run: the measurement behind
``features.rankings_publication_lag_days``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import labelled
from padel_predictor.common.schemas import RANKING_FIRST_SEEN_CAP, deduplicate_matches
from padel_predictor.features.pipeline import load_rankings

#: How many of the latest observed rankings the report lists.
RANKING_DELAY_ROWS = 8


def ranking_delays(rankings: pd.DataFrame) -> pd.DataFrame:
    """When each weekly ranking was first seen, and how long after its label date.

    One row per snapshot date: ``first_seen_at`` is the earliest stored
    publication time of any of its players, ``delay_days`` the gap to the label
    date. ``observed`` is true when a live run saw the snapshot within
    :data:`~padel_predictor.common.schemas.RANKING_FIRST_SEEN_CAP` of its date.
    Otherwise it was backfilled, or first read too late, and the stored time
    says nothing about when it appeared.
    """
    columns = ["snapshot_date", "first_seen_at", "delay_days", "observed"]
    if rankings.empty:
        return pd.DataFrame(columns=columns)
    first = (
        rankings.groupby("snapshot_date")["published_at"]
        .min()
        .rename("first_seen_at")
        .reset_index()
        .sort_values("snapshot_date")
        .reset_index(drop=True)
    )
    delay = first["first_seen_at"] - first["snapshot_date"]
    first["delay_days"] = delay.dt.total_seconds() / 86400.0
    first["observed"] = delay < RANKING_FIRST_SEEN_CAP
    return first[columns]


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
    #: :func:`ranking_delays` of the store, and the lag the features assume.
    ranking_delays: pd.DataFrame = field(default_factory=lambda: ranking_delays(pd.DataFrame()))
    publication_lag_days: float | None = None

    def ranking_delay_markdown(self) -> list[str]:
        """The measured publication delay next to the rule the features use."""
        lag = self.publication_lag_days
        rule = (
            "the publication time as stored"
            if lag is None
            else f"{lag:g} days after the date it is labelled with, in a backfill and "
            "in a live run alike"
        )
        lines = [
            "",
            "**When each weekly ranking first appeared.** A ranking counts as published "
            f"{rule} (`features.rankings_publication_lag_days`).",
        ]
        observed = self.ranking_delays.loc[self.ranking_delays["observed"].astype(bool)]
        unknown = len(self.ranking_delays) - len(observed)
        if observed.empty:
            lines.append(
                "No ranking has been seen by a live run within a week of its date yet, "
                "so the delay is not measured."
            )
        else:
            delay = observed["delay_days"]
            lines.append(
                f"Seen by a live run: {len(observed)} ranking(s), first seen "
                f"{delay.min():.1f} to {delay.max():.1f} days after their date "
                f"(median {delay.median():.1f}). The run is daily, so each may have "
                "appeared up to a day earlier."
            )
            if lag is not None and delay.max() > lag:
                lines.append(
                    f"**The rule is too early:** a ranking was first seen {delay.max():.1f} "
                    f"days after its date, later than the {lag:g} days the features assume."
                )
            lines += ["", "| Ranking dated | First seen (UTC) | Days after its date |"]
            lines.append("| --- | --- | ---: |")
            for row in observed.tail(RANKING_DELAY_ROWS).itertuples(index=False):
                lines.append(
                    f"| {row.snapshot_date.date()} | {row.first_seen_at:%Y-%m-%d %H:%M} "
                    f"| {row.delay_days:.1f} |"
                )
        if unknown:
            lines += [
                "",
                f"{unknown} ranking(s) were backfilled or first read a week or more after "
                "their date, so when they appeared is not known.",
            ]
        return lines

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
                "| Baseline on the test set | Matches | Log loss | Accuracy |",
                "| --- | ---: | ---: | ---: |",
            ]
            rated = int(b["baseline_provider_elo_rated_rows"])
            if rated:
                # The bar of the success criterion, and the in-house Elo on the
                # same matches so the two can be compared like for like.
                lines += [
                    f"| **Provider Elo where it has a rating** (as published before each match; "
                    f"{b['baseline_provider_elo_coverage']:.0%} of the test set) "
                    f"| {rated} | **{b['baseline_provider_elo_rated_log_loss']:.4f}** "
                    f"| {b['baseline_provider_elo_rated_accuracy']:.3f} |",
                    f"| In-house Elo, same matches | {rated} "
                    f"| {b['baseline_elo_rated_log_loss']:.4f} "
                    f"| {b['baseline_elo_rated_accuracy']:.3f} |",
                ]
            lines += [
                f"| Provider Elo, all matches (unrated at 0.5) | {self.test_rows} "
                f"| {b['baseline_provider_elo_log_loss']:.4f} "
                f"| {b['baseline_provider_elo_accuracy']:.3f} |",
                f"| In-house Elo, all matches | {self.test_rows} "
                f"| {b['baseline_elo_log_loss']:.4f} | {b['baseline_elo_accuracy']:.3f} |",
                f"| Majority class | {self.test_rows} | {b['baseline_majority_log_loss']:.4f} "
                f"| {b['baseline_majority_accuracy']:.3f} |",
            ]
        if not self.ranking_delays.empty:
            lines += self.ranking_delay_markdown()
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
        ranking_delays=ranking_delays(rankings),
        publication_lag_days=config.features.rankings_publication_lag_days,
    )
