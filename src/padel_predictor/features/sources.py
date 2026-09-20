"""Ingestion adapters.

Every adapter returns a frame satisfying the raw-match contract in
:mod:`padel_predictor.common.schemas`, so the rest of the feature pipeline does
not care where matches came from.

Two adapters ship with the project:

``sample``
    Deterministic synthetic matches. No network, no credentials -- this is what
    CI and a fresh clone run on, and what makes the pipeline reproducible.
``http``
    Reads a live JSON endpoint configured through ``PADEL_API_BASE_URL``.
    :meth:`HttpMatchSource.parse_records` holds the mapping from the endpoint's
    payload to the raw-match schema and is the one place to adapt when pointing
    the pipeline at a different provider.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any, Protocol

import httpx
import pandas as pd

from padel_predictor.common.config import IngestionConfig, Settings
from padel_predictor.common.logging import get_logger
from padel_predictor.common.schemas import (
    RAW_MATCH_COLUMNS,
    empty_raw_matches,
    normalise_raw_matches,
)

logger = get_logger(__name__)


class MatchSource(Protocol):
    """Anything that can return raw matches for a date range."""

    name: str

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Return matches with ``start <= played_at.date() <= end``."""
        ...


# --------------------------------------------------------------------------- #
# Synthetic source
# --------------------------------------------------------------------------- #
#: Players are deliberately generic: this source invents data, and nothing here
#: should be mistaken for a real player's results.
SAMPLE_PLAYER_COUNT = 40
SAMPLE_TIERS = ("major", "p1", "p2", "challenger")
SAMPLE_ROUNDS = ("R32", "R16", "QF", "SF", "F")
#: Tournaments run Wednesday to Sunday.
SAMPLE_MATCH_WEEKDAYS = frozenset({2, 3, 4, 5, 6})
SAMPLE_MATCHES_PER_DAY = 6


def _player_name(index: int) -> str:
    return f"Player {index:02d}"


def _player_strength(index: int) -> float:
    """A fixed latent skill per player, so outcomes are learnable but noisy."""
    # Deterministic pseudo-random spread in roughly [-1, 1].
    return math.sin(index * 12.9898) * 43758.5453 % 2.0 - 1.0


class SampleMatchSource:
    """Deterministic synthetic matches, seeded by date.

    The same date always yields the same matches, which makes backfills
    idempotent and lets tests assert on exact values.
    """

    name = "sample"

    def __init__(self, seed: int = 20260917) -> None:
        self.seed = seed

    def _day_rng(self, day: dt.date) -> Any:
        import numpy as np

        return np.random.default_rng(self.seed + day.toordinal())

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        now = dt.datetime.now(dt.UTC)
        records: list[dict[str, Any]] = []
        day = start
        while day <= end:
            if day.weekday() in SAMPLE_MATCH_WEEKDAYS:
                records.extend(self._matches_for_day(day, now))
            day += dt.timedelta(days=1)

        if not records:
            return empty_raw_matches()
        return normalise_raw_matches(pd.DataFrame(records))

    def _matches_for_day(self, day: dt.date, now: dt.datetime) -> list[dict[str, Any]]:
        rng = self._day_rng(day)
        tier = SAMPLE_TIERS[day.isocalendar().week % len(SAMPLE_TIERS)]
        tournament = f"Synthetic {tier.upper()} week {day.isocalendar().week:02d}"

        out: list[dict[str, Any]] = []
        for slot in range(SAMPLE_MATCHES_PER_DAY):
            players = rng.choice(SAMPLE_PLAYER_COUNT, size=4, replace=False)
            a1, a2, b1, b2 = (int(p) for p in players)
            played_at = dt.datetime.combine(day, dt.time(hour=11 + slot, tzinfo=dt.UTC))

            strength_a = _player_strength(a1) + _player_strength(a2)
            strength_b = _player_strength(b1) + _player_strength(b2)
            p_a_wins = 1.0 / (1.0 + math.exp(-1.6 * (strength_a - strength_b)))

            # A match in the future has no result yet -- that is what inference predicts.
            winner: str | None = None
            if played_at <= now:
                winner = "A" if rng.random() < p_a_wins else "B"

            out.append(
                {
                    "match_id": f"syn-{day.isoformat()}-{slot:02d}",
                    "played_at": played_at,
                    "tournament": tournament,
                    "tier": tier,
                    "round": SAMPLE_ROUNDS[slot % len(SAMPLE_ROUNDS)],
                    "team_a_player_1": _player_name(a1),
                    "team_a_player_2": _player_name(a2),
                    "team_b_player_1": _player_name(b1),
                    "team_b_player_2": _player_name(b2),
                    "team_a_rank_points": round(2000 + 600 * strength_a, 1),
                    "team_b_rank_points": round(2000 + 600 * strength_b, 1),
                    "winner": winner,
                }
            )
        return out


# --------------------------------------------------------------------------- #
# Live HTTP source
# --------------------------------------------------------------------------- #
class HttpMatchSource:
    """Reads matches from a live JSON endpoint.

    The endpoint is expected to accept ``from``/``to`` date query parameters and
    return ``{"matches": [ ... ]}`` (or a bare list). Adapt
    :meth:`parse_records` to the provider actually being used -- it is the only
    part of the codebase that knows the provider's payload shape.
    """

    name = "http"

    def __init__(self, settings: Settings, config: IngestionConfig) -> None:
        if not settings.api_base_url:
            raise ValueError(
                "PADEL_API_BASE_URL is not set; set it in .env (see .env.example) "
                "or use PADEL_SOURCE=sample"
            )
        self.base_url = settings.api_base_url.rstrip("/")
        self.api_key = settings.api_key.get_secret_value()
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        url = f"{self.base_url}/matches"
        params = {"from": start.isoformat(), "to": end.isoformat()}
        transport = httpx.HTTPTransport(retries=self.config.max_retries)

        with httpx.Client(timeout=self.config.timeout_seconds, transport=transport) as client:
            response = client.get(url, params=params, headers=self._headers())
            response.raise_for_status()
            payload = response.json()

        records = payload.get("matches", []) if isinstance(payload, dict) else payload
        logger.info("fetched %d records from %s for %s..%s", len(records), url, start, end)
        return self.parse_records(records)

    @staticmethod
    def parse_records(records: list[dict[str, Any]]) -> pd.DataFrame:
        """Map the provider's payload onto the raw-match schema.

        Adapt the key names below to the live provider. Records missing a key
        are dropped rather than silently filled, so a schema change upstream
        shows up as a row-count drop instead of a column of nulls.
        """
        if not records:
            return empty_raw_matches()

        rows: list[dict[str, Any]] = []
        for record in records:
            try:
                rows.append(
                    {
                        "match_id": str(record["id"]),
                        "played_at": record["scheduled_at"],
                        "tournament": record.get("tournament", {}).get("name", "unknown"),
                        "tier": record.get("tournament", {}).get("tier", "unknown"),
                        "round": record.get("round", "unknown"),
                        "team_a_player_1": record["teams"][0]["players"][0]["name"],
                        "team_a_player_2": record["teams"][0]["players"][1]["name"],
                        "team_b_player_1": record["teams"][1]["players"][0]["name"],
                        "team_b_player_2": record["teams"][1]["players"][1]["name"],
                        "team_a_rank_points": record["teams"][0].get("rank_points"),
                        "team_b_rank_points": record["teams"][1].get("rank_points"),
                        "winner": record.get("winner"),
                    }
                )
            except (KeyError, IndexError, TypeError) as exc:
                logger.warning("skipping malformed record %s: %s", record.get("id", "?"), exc)

        if not rows:
            return empty_raw_matches()
        return normalise_raw_matches(pd.DataFrame(rows, columns=list(RAW_MATCH_COLUMNS)))


def get_source(
    name: str,
    settings: Settings,
    config: IngestionConfig,
) -> MatchSource:
    """Resolve an adapter by name (``PADEL_SOURCE``)."""
    key = name.strip().lower()
    if key == "sample":
        return SampleMatchSource()
    if key == "http":
        return HttpMatchSource(settings, config)
    raise ValueError(f"unknown source {name!r}; expected one of: sample, http")
