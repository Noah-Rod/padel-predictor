"""Ingestion adapters."""

from __future__ import annotations

import datetime as dt

import httpx
import pandas as pd
import pytest

from padel_predictor.common.config import IngestionConfig, Settings
from padel_predictor.common.schemas import RAW_MATCH_COLUMNS
from padel_predictor.features.sources import (
    HttpMatchSource,
    RateLimitExhausted,
    SampleMatchSource,
    get_source,
)


def test_sample_source_is_deterministic():
    window = (dt.date(2026, 3, 1), dt.date(2026, 3, 14))
    first = SampleMatchSource().fetch(*window)
    second = SampleMatchSource().fetch(*window)

    assert not first.empty
    assert first["match_id"].tolist() == second["match_id"].tolist()
    assert first["winner"].tolist() == second["winner"].tolist()


def test_sample_source_respects_the_schema_and_window():
    start, end = dt.date(2026, 3, 1), dt.date(2026, 3, 14)
    matches = SampleMatchSource().fetch(start, end)

    assert list(matches.columns) == list(RAW_MATCH_COLUMNS)
    assert matches["played_at"].dt.date.min() >= start
    assert matches["played_at"].dt.date.max() <= end
    assert matches["match_id"].is_unique


def test_sample_source_leaves_future_matches_unlabelled():
    today = dt.datetime.now(dt.UTC).date()
    matches = SampleMatchSource().fetch(today - dt.timedelta(days=7), today + dt.timedelta(days=7))
    future = matches.loc[matches["played_at"] > dt.datetime.now(dt.UTC)]

    assert not future.empty
    assert future["winner"].isna().all()


def test_sample_source_rejects_a_reversed_window():
    with pytest.raises(ValueError, match="after end"):
        SampleMatchSource().fetch(dt.date(2026, 3, 14), dt.date(2026, 3, 1))


def test_http_source_needs_a_base_url(settings):
    with pytest.raises(ValueError, match="PADEL_API_BASE_URL"):
        HttpMatchSource(settings, IngestionConfig())


def _player(pid: int, name: str) -> dict:
    return {"id": pid, "name": name, "short_name": name, "nationality": "ES", "side": None}


def _match(match_id: int, status: str = "finished", winner: str | None = "team_1", **extra):
    """A match as /tournaments/{id}/matches returns it (fields from the OpenAPI spec)."""
    record = {
        "id": match_id,
        "category": "men",
        "draw": "main",
        "round": 2,
        "round_name": "Semifinals",
        "played_at": "2026-03-07",
        "scheduled_at": "2026-03-07T18:30:00+01:00",
        "status": status,
        "winner": winner,
        "players": {
            "team_1": [_player(1, "Alex Uno"), _player(2, "Bea Dos")],
            "team_2": [_player(3, "Carlos Tres"), _player(4, "Dani Cuatro")],
        },
    }
    record.update(extra)
    return record


TOURNAMENT = {"id": 7, "name": "Test P1", "level": "p1", "status": "finished"}


def test_padelapi_matches_are_mapped_onto_the_schema():
    out = HttpMatchSource.parse_records([{**_match(42), "tournament": TOURNAMENT}])

    assert len(out) == 1
    row = out.iloc[0]
    assert row["match_id"] == "42"
    assert row["played_at"] == pd.Timestamp("2026-03-07T17:30:00Z")
    assert (row["tournament"], row["tier"], row["round"]) == ("Test P1", "p1", "SF")
    # Sides are re-assigned by hash, so check the winner through the pair, not the side.
    winners = (
        {row["team_a_player_1"], row["team_a_player_2"]}
        if row["winner"] == "A"
        else {row["team_b_player_1"], row["team_b_player_2"]}
    )
    assert winners == {"Alex Uno", "Bea Dos"}


def test_a_scheduled_match_is_an_unlabelled_fixture():
    out = HttpMatchSource.parse_records([_match(1, status="scheduled", winner=None)])
    assert out["winner"].isna().all()
    assert out["tier"].tolist() == ["unknown"]


def test_naive_times_are_server_time_and_a_missing_time_falls_back_to_the_day():
    out = HttpMatchSource.parse_records(
        [
            _match(1, scheduled_at="2026-07-01T10:00:00"),
            _match(2, scheduled_at=None),
        ]
    ).set_index("match_id")
    assert out.loc["1", "played_at"] == pd.Timestamp("2026-07-01T08:00:00Z")  # CEST
    assert out.loc["2", "played_at"] == pd.Timestamp("2026-03-07T00:00:00Z")


def test_unplayed_hidden_and_undrawn_matches_are_dropped_not_filled():
    records = [
        _match(1, status="walkover"),
        _match(2, status="bye", winner=None),
        _match(3, status="not_played", winner=None),
        # The free plan hides results older than six months: not a fixture either.
        _match(4, winner="hidden_free_plan"),
        # A later round whose pairs are not decided yet.
        _match(5, status="scheduled", winner=None, players={"team_1": [], "team_2": []}),
        {"id": 6},  # malformed
        _match(7, status="retired", winner="team_2"),
    ]
    out = HttpMatchSource.parse_records(records)
    assert out["match_id"].tolist() == ["7"]
    assert out["winner"].notna().all()


def test_empty_http_payload_returns_the_empty_schema():
    assert HttpMatchSource.parse_records([]).empty


def test_source_registry(settings):
    assert isinstance(get_source("sample", settings, IngestionConfig()), SampleMatchSource)
    with pytest.raises(ValueError, match="unknown source"):
        get_source("carrier-pigeon", settings, IngestionConfig())


def test_http_source_resolves_when_configured(monkeypatch):
    monkeypatch.setenv("PADEL_API_BASE_URL", "https://example.test/api")
    source = get_source("http", Settings(), IngestionConfig())
    assert isinstance(source, HttpMatchSource)
    assert source.base_url == "https://example.test/api"


def test_sample_rankings_are_published_monday_before_the_week_s_matches():
    rankings = SampleMatchSource().fetch_rankings(dt.date(2026, 3, 1), dt.date(2026, 3, 31))
    assert not rankings.empty
    assert (rankings["published_at"].dt.weekday == 0).all()
    assert rankings["provider_elo"].notna().all()


def _ranking(name: str, ranking_type: str, points, date: str = "2026-01-05", **extra):
    return {
        "id": 1,
        "name": name,
        "category": "men",
        "type": ranking_type,
        "ranking": 1,
        "points": points,
        "date": date,
        **extra,
    }


def test_backfilled_snapshot_without_publication_time_counts_as_published_a_week_later():
    fetched_at = pd.Timestamp("2026-06-01T00:00Z")
    out = HttpMatchSource.parse_ranking_records(
        [_ranking("A1", "official", 900), _ranking("A1", "elo", 1610)],
        fetched_at=fetched_at,
        publication_lag=pd.Timedelta(days=7),
    )
    assert len(out) == 1  # official points and Elo merged into one row
    row = out.iloc[0]
    assert row["snapshot_date"] == pd.Timestamp("2026-01-05", tz="UTC")
    assert row["published_at"] == pd.Timestamp("2026-01-12", tz="UTC")
    assert (row["rank_points"], row["provider_elo"]) == (900, 1610)


def test_explicit_publication_time_wins_and_lag_never_passes_the_fetch_time():
    fetched_at = pd.Timestamp("2026-01-07T09:00Z")
    out = HttpMatchSource.parse_ranking_records(
        [
            _ranking("A1", "official", 900, published_at="2026-01-05T10:00Z"),
            _ranking("A2", "official", 800),
        ],
        fetched_at=fetched_at,
    )
    by_player = out.set_index("player")["published_at"]
    assert by_player["A1"] == pd.Timestamp("2026-01-05T10:00Z")
    assert by_player["A2"] == fetched_at


def test_hidden_ranking_values_become_missing_and_race_is_ignored():
    out = HttpMatchSource.parse_ranking_records(
        [_ranking("A1", "official", "hidden_free_plan"), _ranking("A1", "race", 50)],
        fetched_at=pd.Timestamp("2026-06-01T00:00Z"),
    )
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["rank_points"])


# --------------------------------------------------------------------------- #
# The client against a stand-in for padelapi.org
# --------------------------------------------------------------------------- #
class FakePadelApi:
    """Serves /tournaments, /tournaments/{id}/matches and /rankings in pages."""

    #: The players of every fake match, by provider id (see :func:`_match`).
    PLAYERS = {1: "Alex Uno", 2: "Bea Dos", 3: "Carlos Tres", 4: "Dani Cuatro"}

    def __init__(
        self,
        pages_of_matches: int = 2,
        fail_first: dict | None = None,
        unknown_players: frozenset[int] = frozenset(),
    ):
        self.requests: list[httpx.Request] = []
        self.pages_of_matches = pages_of_matches
        self.fail_first = dict(fail_first or {})
        self.unknown_players = unknown_players

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path, params = request.url.path, request.url.params
        for prefix, response in list(self.fail_first.items()):
            if path.startswith(prefix):
                del self.fail_first[prefix]
                return response
        page = int(params.get("page", 1))
        if path == "/api/tournaments":
            return self._page([TOURNAMENT, {**TOURNAMENT, "id": 8, "status": "cancelled"}], 1)
        if path == "/api/tournaments/7/matches":
            match = _match(100 + page, played_at=f"2026-03-0{page + 4}", scheduled_at=None)
            return self._page([match], self.pages_of_matches, page)
        if path == "/api/rankings":
            points = 1600 if params["type"] == "elo" else 900
            date = params.get("date", "2026-03-02")
            return self._page(
                [_ranking(f"P {params['category']}", params["type"], points, date)], 9
            )
        if path.startswith("/api/players/") and path.endswith("/rankings"):
            pid = int(path.split("/")[3])
            if pid in self.unknown_players:
                return httpx.Response(404, json={"message": "Not found."})
            name = self.PLAYERS[pid]
            # A bare array, as the spec describes it, with a type the pipeline ignores.
            return httpx.Response(
                200,
                json=[
                    _ranking(name, "official", 300 + pid, "2026-03-02"),
                    _ranking(name, "elo", 1400 + pid, "2026-03-02"),
                    _ranking(name, "race", 10, "2026-03-02"),
                ],
            )
        return httpx.Response(404, json={"message": "Not found."})

    @staticmethod
    def _page(data: list, last_page: int, page: int = 1) -> httpx.Response:
        return httpx.Response(
            200, json={"data": data, "meta": {"current_page": page, "last_page": last_page}}
        )


def _source(
    monkeypatch, api: FakePadelApi, plan: str = "paid", sleeps: list | None = None, **ingestion
):
    monkeypatch.setenv("PADEL_API_BASE_URL", "https://padelapi.test/api")
    monkeypatch.setenv("PADEL_API_KEY", "token-123")
    monkeypatch.setenv("PADEL_API_PLAN", plan)
    clock = iter(range(0, 10_000, 1))  # one second passes between requests
    return HttpMatchSource(
        Settings(),
        IngestionConfig(requests_per_minute={"free": 10, "paid": 60}, **ingestion),
        transport=httpx.MockTransport(api),
        sleep=(sleeps.append if sleeps is not None else lambda _: None),
        clock=lambda: float(next(clock)),
    )


def test_fetch_reads_every_page_of_each_tournament_and_skips_cancelled_ones(monkeypatch):
    api = FakePadelApi(pages_of_matches=2)
    out = _source(monkeypatch, api).fetch(dt.date(2026, 3, 1), dt.date(2026, 3, 31))

    assert sorted(out["match_id"]) == ["101", "102"]
    paths = [r.url.path for r in api.requests]
    assert "/api/tournaments/8/matches" not in paths
    first = api.requests[0]
    assert first.headers["Authorization"] == "Bearer token-123"
    assert first.url.params["level"] == "major,p1,p2,finals,fip_platinum,fip_gold"
    assert first.url.params["per_page"] == "50"


def test_fetch_keeps_only_matches_inside_the_window(monkeypatch):
    out = _source(monkeypatch, FakePadelApi(pages_of_matches=2)).fetch(
        dt.date(2026, 3, 6), dt.date(2026, 3, 6)
    )
    assert out["match_id"].tolist() == ["102"]


def test_requests_are_paced_to_the_plan_limit(monkeypatch):
    sleeps: list[float] = []
    _source(monkeypatch, FakePadelApi(), plan="free", sleeps=sleeps).fetch(
        dt.date(2026, 3, 1), dt.date(2026, 3, 31)
    )
    # 10 per minute = one every 6 s; the fake clock advances 1 s per call.
    assert sleeps and all(s == pytest.approx(5.0) for s in sleeps)


def test_a_rate_limit_waits_for_retry_after_then_retries(monkeypatch):
    api = FakePadelApi(
        fail_first={"/api/tournaments": httpx.Response(429, headers={"Retry-After": "30"})}
    )
    sleeps: list[float] = []
    out = _source(monkeypatch, api, sleeps=sleeps).fetch(dt.date(2026, 3, 1), dt.date(2026, 3, 31))
    assert 30.0 in sleeps
    assert not out.empty


def test_an_exhausted_daily_quota_stops_the_run(monkeypatch):
    api = FakePadelApi(
        fail_first={"/api/tournaments": httpx.Response(429, headers={"Retry-After": "40000"})}
    )
    with pytest.raises(RateLimitExhausted):
        _source(monkeypatch, api).fetch(dt.date(2026, 3, 1), dt.date(2026, 3, 31))


def test_paid_rankings_read_one_snapshot_per_monday_and_cap_the_pages(monkeypatch):
    api = FakePadelApi()
    out = _source(monkeypatch, api).fetch_rankings(dt.date(2026, 3, 1), dt.date(2026, 3, 10))

    dates = sorted({r.url.params["date"] for r in api.requests})
    assert dates == ["2026-02-23", "2026-03-02", "2026-03-09"]
    # 3 weeks x (official, elo) x (men, women) x 4 pages
    assert len(api.requests) == 3 * 2 * 2 * 4
    assert set(out["player"]) == {"P men", "P women"}
    assert out["provider_elo"].eq(1600).all() and out["rank_points"].eq(900).all()


def test_free_rankings_are_the_latest_snapshot_published_now(monkeypatch):
    api = FakePadelApi()
    before = pd.Timestamp.now(tz="UTC")
    out = _source(monkeypatch, api, plan="free").fetch_rankings(
        dt.date(2026, 1, 1), dt.date(2026, 3, 10)
    )
    assert all("date" not in r.url.params for r in api.requests)
    assert (out["published_at"] >= before).all()


def _player_requests(api: FakePadelApi) -> list[httpx.Request]:
    return [r for r in api.requests if r.url.path.startswith("/api/players/")]


def test_players_missing_from_the_lists_are_read_on_their_own(monkeypatch):
    api = FakePadelApi()
    source = _source(monkeypatch, api)
    source.fetch(dt.date(2026, 3, 1), dt.date(2026, 3, 31))

    out = source.fetch_rankings(dt.date(2026, 3, 1), dt.date(2026, 3, 10))

    # The fake lists only hold "P men" and "P women": all four players are missing.
    requests = _player_requests(api)
    assert sorted(r.url.path for r in requests) == [
        f"/api/players/{pid}/rankings" for pid in (1, 2, 3, 4)
    ]
    assert {(r.url.params["after_date"], r.url.params["before_date"]) for r in requests} == {
        ("2026-03-01", "2026-03-10")
    }
    alex = out.set_index("player").loc["Alex Uno"]
    assert (alex["rank_points"], alex["provider_elo"]) == (301, 1401)
    assert alex["published_at"] == pd.Timestamp("2026-03-09", tz="UTC")  # label + 7 days


def test_a_player_on_both_lists_is_not_read_again(monkeypatch):
    api = FakePadelApi()
    source = _source(monkeypatch, api)
    source.player_ids = {"P men": 99, "Alex Uno": 1}

    source.fetch_rankings(dt.date(2026, 3, 1), dt.date(2026, 3, 10))

    assert [r.url.path for r in _player_requests(api)] == ["/api/players/1/rankings"]


def test_player_reads_are_capped_and_a_player_without_rankings_is_skipped(monkeypatch):
    api = FakePadelApi(unknown_players=frozenset({1}))
    source = _source(monkeypatch, api, rankings_max_player_requests=2)
    source.player_ids = dict(map(reversed, FakePadelApi.PLAYERS.items()))

    out = source.fetch_rankings(dt.date(2026, 3, 1), dt.date(2026, 3, 10))

    # Sorted by name, the first two are Alex Uno (unknown, 404) and Bea Dos.
    assert len(_player_requests(api)) == 2
    assert "Alex Uno" not in set(out["player"])
    assert "Bea Dos" in set(out["player"])


def test_the_free_plan_and_the_switch_never_read_players_on_their_own(monkeypatch):
    for kwargs in ({"plan": "free"}, {"rankings_fill_missing_players": False}):
        api = FakePadelApi()
        source = _source(monkeypatch, api, **kwargs)
        source.player_ids = {"Alex Uno": 1}
        source.fetch_rankings(dt.date(2026, 3, 1), dt.date(2026, 3, 10))
        assert not _player_requests(api)


def test_http_source_rejects_an_unknown_plan(monkeypatch):
    monkeypatch.setenv("PADEL_API_BASE_URL", "https://example.test/api")
    monkeypatch.setenv("PADEL_API_PLAN", "gold")
    with pytest.raises(ValueError, match="PADEL_API_PLAN"):
        HttpMatchSource(Settings(), IngestionConfig())
