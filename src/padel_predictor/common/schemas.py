"""Data contracts.

Every ingestion adapter must produce a frame with :data:`RAW_MATCH_COLUMNS`.
Keeping the contract in one place means a new data source cannot silently
change the shape the rest of the system depends on.
"""

from __future__ import annotations

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
    "team_a_rank_points",  # combined ranking points of team A at match time
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

    return out.sort_values(["played_at", "match_id"], kind="stable").reset_index(drop=True)


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
    return out.sort_values(["played_at", "match_id"], kind="stable").reset_index(drop=True)
