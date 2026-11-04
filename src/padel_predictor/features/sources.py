"""Ingestion adapters.

Every adapter returns a frame satisfying the raw-match contract in
:mod:`padel_predictor.common.schemas`, so the rest of the feature pipeline does
not care where matches came from.

Two adapters ship with the project:

``sample``
    Deterministic synthetic matches. No network, no credentials -- this is what
    CI and a fresh clone run on, and what makes the pipeline reproducible.
``http``
    Reads padelapi.org (``PADEL_API_BASE_URL=https://padelapi.org/api``) with a
    bearer token. :meth:`HttpMatchSource.parse_records` and
    :meth:`HttpMatchSource.parse_ranking_records` hold the whole mapping from
    the provider's payload to the raw schemas.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx
import pandas as pd

from padel_predictor.common.config import IngestionConfig, Settings
from padel_predictor.common.logging import get_logger
from padel_predictor.common.schemas import (
    RAW_MATCH_COLUMNS,
    RAW_RANKING_COLUMNS,
    empty_raw_matches,
    empty_raw_rankings,
    normalise_raw_matches,
    normalise_raw_rankings,
    orient_teams,
)

logger = get_logger(__name__)


class MatchSource(Protocol):
    """Anything that can return raw matches for a date range."""

    name: str

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Return matches with ``start <= played_at.date() <= end``."""
        ...

    def fetch_rankings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Return ranking snapshots labelled ``start <= snapshot_date <= end``."""
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
#: Synthetic rankings are labelled with the Monday of each week and published
#: that Monday morning, before the Wednesday start of the week's tournament.
SAMPLE_RANKING_PUBLISH_TIME = dt.time(hour=8, tzinfo=dt.UTC)


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

    def fetch_rankings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Weekly snapshots: official points and a provider Elo per player.

        The provider Elo is a noisy view of the same latent strength the match
        outcomes are drawn from, so the synthetic baseline is good but beatable.
        """
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        now = dt.datetime.now(dt.UTC)
        monday = start - dt.timedelta(days=start.weekday())
        if monday < start:
            monday += dt.timedelta(days=7)

        records: list[dict[str, Any]] = []
        while monday <= end:
            published_at = dt.datetime.combine(monday, SAMPLE_RANKING_PUBLISH_TIME)
            if published_at <= now:
                rng = self._day_rng(monday)
                for index in range(SAMPLE_PLAYER_COUNT):
                    strength = _player_strength(index)
                    records.append(
                        {
                            "player": _player_name(index),
                            "snapshot_date": monday,
                            "published_at": published_at,
                            "rank_points": round(1000 + 300 * strength + rng.normal(0, 40), 1),
                            "provider_elo": round(1500 + 180 * strength + rng.normal(0, 60), 1),
                        }
                    )
            monday += dt.timedelta(days=7)

        if not records:
            return empty_raw_rankings()
        return normalise_raw_rankings(pd.DataFrame(records))

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
                    # Ranking points come from the weekly snapshots, not the match.
                    "team_a_rank_points": None,
                    "team_b_rank_points": None,
                    "winner": winner,
                }
            )
        return out


# --------------------------------------------------------------------------- #
# Live HTTP source
# --------------------------------------------------------------------------- #
#: padelapi.org's ``round`` is the number of matches in the round; these are the
#: labels the rest of the system uses.
PADELAPI_ROUND_LABELS: dict[int, str] = {
    1: "F",
    2: "SF",
    4: "QF",
    8: "R16",
    16: "R32",
    32: "R64",
    0: "G",
}
#: Statuses whose ``winner`` is a played result. ``retired`` started and has a
#: winner; the provider's own Elo counts it too.
LABELLED_STATUSES = frozenset({"finished", "retired"})
#: Statuses that are not a played match: no row, labelled or upcoming.
DROPPED_STATUSES = frozenset({"walkover", "bye", "not_played"})
#: What the free plan returns instead of values it does not include.
HIDDEN = "hidden_free_plan"
#: Timestamps without an offset are in the provider's server time zone.
SERVER_TIMEZONE = "Europe/Madrid"
RANKING_TYPES = {"official": "rank_points", "elo": "provider_elo"}
CATEGORIES = ("men", "women")


class RateLimitExhausted(RuntimeError):
    """The daily (or monthly) request quota is used up; resume the run later."""


class _PadelApiClient:
    """GET with pagination, pacing to the plan's per-minute limit and 429 handling."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        config: IngestionConfig,
        requests_per_minute: int,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=config.timeout_seconds,
            # Merged players, tournaments and matches answer with a 302 to the
            # canonical record.
            follow_redirects=True,
            transport=transport or httpx.HTTPTransport(retries=config.max_retries),
        )
        self.config = config
        self.min_interval = 60.0 / max(requests_per_minute, 1)
        self._sleep = sleep
        self._clock = clock
        self._last_request: float | None = None
        self.requests = 0

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> _PadelApiClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _pace(self) -> None:
        if self._last_request is not None:
            wait = self.min_interval - (self._clock() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        self._last_request = self._clock()

    def get(self, path: str, params: dict[str, Any]) -> Any:
        for attempt in range(self.config.max_retries + 1):
            self._pace()
            self.requests += 1
            response = self._http.get(path, params=params)
            if response.status_code == 429:
                retry_after = float(response.headers.get("Retry-After", 60))
                if retry_after > self.config.max_retry_after_seconds:
                    raise RateLimitExhausted(
                        f"padelapi.org quota exhausted (Retry-After {retry_after:.0f}s); "
                        "re-run later, a backfill resumes from its last chunk"
                    )
                logger.warning("rate limited on %s, waiting %.0fs", path, retry_after)
                self._sleep(retry_after)
                continue
            if response.status_code >= 500 and attempt < self.config.max_retries:
                self._sleep(2.0**attempt)
                continue
            if response.status_code == 402:
                raise PermissionError(
                    f"{path} is not included in the current padelapi.org plan; check PADEL_API_PLAN"
                )
            response.raise_for_status()
            return response.json()
        raise RuntimeError(f"giving up on {path} after {self.config.max_retries + 1} attempts")

    def paginate(
        self, path: str, params: dict[str, Any], max_pages: int | None = None
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self.get(path, {**params, "page": page, "per_page": self.config.page_size})
            data = payload.get("data", []) if isinstance(payload, dict) else payload
            records.extend(data)
            meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
            last_page = int(meta.get("last_page") or page)
            if not data or page >= last_page or (max_pages is not None and page >= max_pages):
                return records
            page += 1


def _start_time(record: dict[str, Any]) -> pd.Timestamp | None:
    """The match start in UTC: ``scheduled_at``, else the day it is played."""
    scheduled = record.get("scheduled_at")
    if scheduled and scheduled != HIDDEN:
        ts = pd.Timestamp(scheduled)
        ts = ts.tz_localize(SERVER_TIMEZONE) if ts.tzinfo is None else ts
        return ts.tz_convert("UTC")
    day = record.get("played_at")
    if day and day != HIDDEN:
        # Midnight is the earliest the match can start, so the as-of ranking
        # join can only ever be stricter than with the real start time.
        return pd.Timestamp(day).tz_localize("UTC")
    return None


def _pair(players: Any) -> tuple[str, str] | None:
    """The two player names of a team, or ``None`` while the pair is not known."""
    if not isinstance(players, list) or len(players) != 2:
        return None
    names = [p.get("name") for p in players if isinstance(p, dict)]
    if len(names) != 2 or not all(names):
        return None
    return names[0], names[1]


def _mondays(start: dt.date, end: dt.date) -> list[dt.date]:
    first = start - dt.timedelta(days=start.weekday())
    return [first + dt.timedelta(weeks=i) for i in range((end - first).days // 7 + 1)]


class HttpMatchSource:
    """Reads matches and rankings from padelapi.org (``PADEL_API_BASE_URL``).

    Matches are read tournament by tournament, as the provider recommends:
    ``/tournaments`` lists the editions running in the window at the levels in
    ``ingestion.levels``, then ``/tournaments/{id}/matches`` pages through each
    one's main draw. Ranking snapshots come from ``/rankings`` per ISO week
    (paid plan) or as the latest snapshot only (free plan).
    """

    name = "http"

    def __init__(
        self,
        settings: Settings,
        config: IngestionConfig,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.plan = settings.api_plan.strip().lower()
        if self.plan not in {"paid", "free"}:
            raise ValueError(f"PADEL_API_PLAN must be 'paid' or 'free', got {settings.api_plan!r}")
        if not settings.api_base_url:
            raise ValueError(
                "PADEL_API_BASE_URL is not set; set it in .env (see .env.example) "
                "or use PADEL_SOURCE=sample"
            )
        self.base_url = settings.api_base_url.rstrip("/")
        self.api_key = settings.api_key.get_secret_value()
        self.config = config
        self._transport = transport
        self._sleep = sleep
        self._clock = clock

    def _client(self) -> _PadelApiClient:
        return _PadelApiClient(
            self.base_url,
            self.api_key,
            self.config,
            requests_per_minute=self.config.requests_per_minute.get(self.plan, 10),
            transport=self._transport,
            sleep=self._sleep,
            clock=self._clock,
        )

    def fetch(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        if start > end:
            raise ValueError(f"start {start} is after end {end}")

        records: list[dict[str, Any]] = []
        with self._client() as client:
            tournaments = client.paginate(
                "/tournaments",
                {
                    "after_date": start.isoformat(),
                    "before_date": end.isoformat(),
                    "level": ",".join(self.config.levels),
                    "format": "pairs",
                    "sort_by": "start_date",
                    "order_by": "asc",
                },
            )
            for tournament in tournaments:
                if tournament.get("status") == "cancelled":
                    continue
                matches = client.paginate(
                    f"/tournaments/{tournament['id']}/matches",
                    {"sort_by": "played_at", "order_by": "asc"},
                )
                records.extend({**match, "tournament": tournament} for match in matches)
            logger.info(
                "fetched %d matches from %d tournaments for %s..%s (%d requests)",
                len(records),
                len(tournaments),
                start,
                end,
                client.requests,
            )

        frame = self.parse_records(records)
        days = frame["played_at"].dt.date
        return frame.loc[(days >= start) & (days <= end)].reset_index(drop=True)

    def fetch_rankings(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Ranking snapshots for a window.

        The paid plan reads one weekly snapshot per Monday in the window. The
        free plan only shows the latest snapshot, which is stored as published
        *now*: it therefore only ever joins onto matches that start after this
        run, never onto history.
        """
        if start > end:
            raise ValueError(f"start {start} is after end {end}")
        now = pd.Timestamp.now(tz="UTC")
        today = now.date()
        # The free plan hides every snapshot but the latest one.
        dates: list[dt.date | None] = (
            [None] if self.plan == "free" else list(_mondays(start, min(end, today)))
        )

        records: list[dict[str, Any]] = []
        with self._client() as client:
            for date in dates:
                for ranking_type in RANKING_TYPES:
                    for category in CATEGORIES:
                        params: dict[str, Any] = {
                            "type": ranking_type,
                            "category": category,
                            "sort_by": "ranking",
                            "order_by": "asc",
                        }
                        if date is not None:
                            params["date"] = date.isoformat()
                        records.extend(
                            client.paginate(
                                "/rankings", params, max_pages=self.config.rankings_max_pages
                            )
                        )
            logger.info(
                "fetched %d ranking entries for %d snapshot(s) (%d requests)",
                len(records),
                len(dates),
                client.requests,
            )

        if self.plan == "free":
            records = [{**r, "published_at": now.isoformat()} for r in records]
        return self.parse_ranking_records(
            records,
            fetched_at=now,
            publication_lag=pd.Timedelta(days=self.config.rankings_publication_lag_days),
        )

    @staticmethod
    def parse_ranking_records(
        records: list[dict[str, Any]],
        fetched_at: pd.Timestamp,
        publication_lag: pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Map ``/rankings`` entries onto the ranking schema, one row per player and week.

        An ``official`` entry carries the FIP points, an ``elo`` entry carries
        the provider's Elo in ``points``; both are merged on (player, date).

        ``published_at`` is the join key, so it is never guessed optimistically:
        the provider's own publication time when there is one, otherwise the
        label date plus ``publication_lag``, and never later than ``fetched_at``
        would make it -- a snapshot cannot have been available after we saw it,
        but it may have been available later than its label suggests.
        """
        if not records:
            return empty_raw_rankings()
        publication_lag = pd.Timedelta(days=7) if publication_lag is None else publication_lag
        rows: dict[tuple[str, pd.Timestamp], dict[str, Any]] = {}
        for record in records:
            try:
                column = RANKING_TYPES.get(record.get("type", "official"))
                if column is None:
                    continue
                name = record["name"]
                label_ts = pd.Timestamp(record["date"])
                label_ts = label_ts.tz_localize("UTC") if label_ts.tzinfo is None else label_ts
                published = record.get("published_at")
                if published:
                    published_ts = pd.Timestamp(published)
                else:
                    published_ts = min(label_ts + publication_lag, fetched_at)
                points = record.get("points")
                row = rows.setdefault(
                    (name, label_ts),
                    {
                        "player": name,
                        "snapshot_date": label_ts,
                        "published_at": published_ts,
                        "rank_points": None,
                        "provider_elo": None,
                    },
                )
                row[column] = None if points == HIDDEN else points
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("skipping malformed ranking record: %s", exc)
        if not rows:
            return empty_raw_rankings()
        frame = pd.DataFrame(list(rows.values()), columns=list(RAW_RANKING_COLUMNS))
        return normalise_raw_rankings(frame)

    @staticmethod
    def parse_records(records: list[dict[str, Any]]) -> pd.DataFrame:
        """Map padelapi.org matches (each with its ``tournament`` embedded) onto the schema.

        Kept: ``finished`` and ``retired`` with their winner, and every other
        status whose two pairs are known as an unlabelled fixture. Dropped: a
        walkover, bye or ``not_played``, a match whose pairs are not drawn yet,
        and a finished match whose winner the free plan hides -- it is neither
        a label nor a fixture. Records missing a key are dropped rather than
        silently filled, so a schema change upstream shows up as a row-count
        drop instead of a column of nulls.
        """
        if not records:
            return empty_raw_matches()

        rows: list[dict[str, Any]] = []
        skipped = {"not played": 0, "pairs unknown": 0, "winner hidden": 0, "no date": 0}
        for record in records:
            try:
                status = record["status"]
                if status in DROPPED_STATUSES:
                    skipped["not played"] += 1
                    continue
                team_1 = _pair(record["players"]["team_1"])
                team_2 = _pair(record["players"]["team_2"])
                if team_1 is None or team_2 is None:
                    skipped["pairs unknown"] += 1
                    continue
                winner = None
                if status in LABELLED_STATUSES:
                    if record.get("winner") not in ("team_1", "team_2"):
                        skipped["winner hidden"] += 1
                        continue
                    winner = "A" if record["winner"] == "team_1" else "B"
                played_at = _start_time(record)
                if played_at is None:
                    skipped["no date"] += 1
                    continue
                tournament = record.get("tournament") or {}
                round_code = record.get("round")
                rows.append(
                    {
                        "match_id": str(record["id"]),
                        "played_at": played_at,
                        "tournament": tournament.get("name", "unknown"),
                        "tier": tournament.get("level", "unknown"),
                        "round": PADELAPI_ROUND_LABELS.get(round_code, str(round_code)),
                        "team_a_player_1": team_1[0],
                        "team_a_player_2": team_1[1],
                        "team_b_player_1": team_2[0],
                        "team_b_player_2": team_2[1],
                        "team_a_rank_points": None,
                        "team_b_rank_points": None,
                        "winner": winner,
                    }
                )
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                logger.warning("skipping malformed record %s: %s", record.get("id", "?"), exc)
        if any(skipped.values()):
            logger.info("skipped matches: %s", skipped)

        if not rows:
            return empty_raw_matches()
        frame = normalise_raw_matches(pd.DataFrame(rows, columns=list(RAW_MATCH_COLUMNS)))
        # team_1 is the higher seed; sides are re-assigned by a hash of the pair names.
        return orient_teams(frame)


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
