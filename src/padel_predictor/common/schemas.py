"""Data contracts.

Every ingestion adapter must produce a frame with :data:`RAW_MATCH_COLUMNS`.
Keeping the contract in one place means a new data source cannot silently
change the shape the rest of the system depends on.
"""

from __future__ import annotations

import hashlib

import pandas as pd

#: Columns every ingestion adapter must return, in order.
RAW_MATCH_COLUMNS: tuple[str, ...] = (
    "match_id",  # stable unique id from the source
    "played_at",  # UTC timestamp of the match start
    "tournament",  # tournament name
    "tier",  # tour tier, e.g. "major" / "p1" / "p2" / "challenger"
    "round",  # e.g. "R16", "QF", "SF", "F"
    "team_a_player_1",
    "team_a_player_2",
    "team_b_player_1",
    "team_b_player_2",
    # Combined ranking points of team A. Only used when no ranking history is
    # passed to the feature builder; the pipelines always pass one, so these
    # columns are overwritten by the as-of join on publication time.
    "team_a_rank_points",
    "team_b_rank_points",
    "winner",  # "A", "B", or <NA> for a match that has not been played yet
)

#: Player-name columns, team A first.
PLAYER_COLUMNS: tuple[str, ...] = (
    "team_a_player_1",
    "team_a_player_2",
    "team_b_player_1",
    "team_b_player_2",
)

VALID_WINNERS: frozenset[str] = frozenset({"A", "B"})

#: Draw order of the round labels. Matches without a start time share their
#: day's midnight, and a pair plays several rounds of one event in a few days,
#: so on equal ``played_at`` the earlier round has to be replayed first.
ROUND_ORDER: dict[str, int] = {"G": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}


def chronological(df: pd.DataFrame) -> pd.DataFrame:
    """Sort matches by start time, then round, then ``match_id``."""
    round_rank = df["round"].map(ROUND_ORDER).fillna(-1) if "round" in df else 0
    return (
        df.assign(_round_rank=round_rank)
        .sort_values(["played_at", "_round_rank", "match_id"], kind="stable")
        .drop(columns="_round_rank")
        .reset_index(drop=True)
    )


class SchemaError(ValueError):
    """Raised when a frame does not satisfy the raw-match contract."""


def empty_raw_matches() -> pd.DataFrame:
    """An empty frame with the raw-match schema and correct dtypes."""
    return normalise_raw_matches(pd.DataFrame(columns=list(RAW_MATCH_COLUMNS)))


def normalise_raw_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a raw-match frame into the canonical schema.

    Raises:
        SchemaError: if required columns are missing or ``winner`` holds a value
            other than ``"A"``, ``"B"`` or null.
    """
    missing = [c for c in RAW_MATCH_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"raw matches missing required columns: {missing}")

    out = df.loc[:, list(RAW_MATCH_COLUMNS)].copy()
    out["match_id"] = out["match_id"].astype("string")
    out["played_at"] = pd.to_datetime(out["played_at"], utc=True, errors="coerce")
    for col in ("tournament", "tier", "round", *PLAYER_COLUMNS):
        out[col] = out[col].astype("string").str.strip()
    for col in ("team_a_rank_points", "team_b_rank_points"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Float64")

    out["winner"] = out["winner"].astype("string").str.upper().str.strip()
    out.loc[~out["winner"].isin(VALID_WINNERS), "winner"] = pd.NA

    if out["played_at"].isna().any():
        bad = out.loc[out["played_at"].isna(), "match_id"].tolist()
        raise SchemaError(f"matches with unparseable played_at: {bad[:5]}")
    if out["match_id"].isna().any():
        raise SchemaError("matches with a null match_id")

    return chronological(out)


def deduplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per ``match_id``, preferring the most recently ingested one.

    The feature pipeline re-reads an overlapping window on every run, so the same
    match arrives many times -- and a match that was upcoming at one run is
    finished at a later one. Sorting by ``ingested_at`` before dropping duplicates
    keeps the newest (and therefore labelled) version.
    """
    if df.empty:
        return df
    out = df.copy()
    if "ingested_at" in out.columns:
        out = out.sort_values("ingested_at", kind="stable")
    out = out.drop_duplicates(subset="match_id", keep="last")
    return chronological(out)


def orient_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Assign sides A and B by a hash of the two pair names, never by the source's order.

    Providers list the higher seed first. Keeping that order would make "team A
    wins" mostly true and let a model score well by always guessing A. The side
    is derived from a hash of the two (order-independent) pair keys, so the
    label is balanced by construction and re-ingesting a match always yields
    the same orientation.
    """
    if df.empty:
        return df
    out = df.copy()
    a_cols = ["team_a_player_1", "team_a_player_2", "team_a_rank_points"]
    b_cols = ["team_b_player_1", "team_b_player_2", "team_b_rank_points"]

    def needs_swap(row: pd.Series) -> bool:
        pair_a = "|".join(sorted(str(row[c]) for c in a_cols[:2]))
        pair_b = "|".join(sorted(str(row[c]) for c in b_cols[:2]))
        first, second = sorted((pair_a, pair_b))
        digest = hashlib.sha1(f"{first}#{second}".encode()).digest()
        # The hash decides whether the alphabetically first pair plays as A.
        first_is_a = digest[0] % 2 == 0
        return (pair_a == first) != first_is_a

    swap = out.apply(needs_swap, axis=1).to_numpy(dtype=bool)
    if swap.any():
        out.loc[swap, a_cols + b_cols] = out.loc[swap, b_cols + a_cols].to_numpy()
        winner = out["winner"].astype("string")
        out.loc[swap, "winner"] = winner[swap].map({"A": "B", "B": "A"})
    return out


# --------------------------------------------------------------------------- #
# Ranking snapshots
# --------------------------------------------------------------------------- #
#: Columns every ranking adapter must return: one row per player per snapshot.
RAW_RANKING_COLUMNS: tuple[str, ...] = (
    "player",
    "snapshot_date",  # the date the provider labels the snapshot with
    "published_at",  # UTC time the snapshot became available -- the join key
    "rank_points",  # official ranking points of the player
    "provider_elo",  # the data provider's own Elo rating of the player
)


def empty_raw_rankings() -> pd.DataFrame:
    """An empty frame with the ranking schema and correct dtypes."""
    return normalise_raw_rankings(pd.DataFrame(columns=list(RAW_RANKING_COLUMNS)))


def normalise_raw_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce a ranking-snapshot frame into the canonical schema.

    Raises:
        SchemaError: if columns are missing or a row has no ``published_at``.
            A snapshot without a publication time cannot be joined safely, so
            adapters must fill it (see ``HttpMatchSource.parse_ranking_records``).
    """
    missing = [c for c in RAW_RANKING_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"raw rankings missing required columns: {missing}")

    out = df.loc[:, list(RAW_RANKING_COLUMNS)].copy()
    out["player"] = out["player"].astype("string").str.strip()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], utc=True, errors="coerce")
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True, errors="coerce")
    for col in ("snapshot_date", "published_at"):
        out[col] = out[col].astype("datetime64[ns, UTC]")
    for col in ("rank_points", "provider_elo"):
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    if out["published_at"].isna().any():
        raise SchemaError("ranking snapshots with an unparseable published_at")
    if out["player"].isna().any():
        raise SchemaError("ranking snapshots with a null player")
    return out.sort_values(["published_at", "player"], kind="stable").reset_index(drop=True)


def deduplicate_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player and snapshot, keeping the *earliest* publication time.

    The daily run re-reads recent snapshots. The first time a snapshot was seen
    is when it was really available, so a later re-read must not move it.
    """
    if df.empty:
        return empty_raw_rankings()
    out = normalise_raw_rankings(df)
    out = out.sort_values("published_at", kind="stable")
    out = out.drop_duplicates(subset=["player", "snapshot_date"], keep="first")
    return out.sort_values(["published_at", "player"], kind="stable").reset_index(drop=True)
