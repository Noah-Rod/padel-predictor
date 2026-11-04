"""Feature definitions -- the single source of truth for the whole system.

Both the training pipeline and the inference pipeline import
:func:`build_feature_frame` from this module. There is no second implementation
of "how a match becomes a feature row" anywhere in the repository, which is what
keeps training and serving from drifting apart.

**No leakage by construction.** Features are derived from rolling state (Elo,
recent form, rest, partnership history) replayed in chronological order. For
every match the feature row is computed *before* that match's result is folded
into the state, so a row only ever sees matches that finished earlier.

**Rankings are joined on publication time.** Official points and the provider's
Elo come from weekly snapshots. A snapshot is matched to a match only if its
``published_at`` is strictly before the match's scheduled start
(:func:`attach_rankings`), never by the date the snapshot is labelled with: a
snapshot labelled with a tournament week can be published after that week and
already contain its results.

**From two players to one team.** Padel is played in pairs, so every
player-level value is aggregated to the pair before the A-minus-B difference:

=====================  =========================================================
feature                team value
=====================  =========================================================
``elo_diff``           mean of the two players' in-house Elo (start 1500, K=24)
``rank_points_diff``   sum of the two players' official points, from the last
                       snapshot published before the match (unranked = 0)
``form_diff``          mean of the two players' win rates over each player's last
                       ``form_window`` (5) matches, with any partner; 0.5 when a
                       player has no history
``rest_days_diff``     days since the pair's most recently active player last
                       played (the smaller gap), capped at 60; 14 for a player
                       never seen
``partnership_diff``   matches this exact pair has played together before
``tier_level``         ordinal tier of the event (not a difference)
=====================  =========================================================

A new partnership has ``partnership = 0``, but it is not treated as a new team:
Elo, form and rest stay per player, so two experienced players who just paired
up keep their individual strength and only the partnership feature is reset.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Sequence

import pandas as pd

from padel_predictor.common.config import FeatureConfig, get_pipeline_config
from padel_predictor.common.schemas import normalise_raw_matches

#: Model inputs, in the order the model expects them.
FEATURE_COLUMNS: tuple[str, ...] = (
    "elo_diff",
    "rank_points_diff",
    "form_diff",
    "rest_days_diff",
    "partnership_diff",
    "tier_level",
)

#: The label: 1 when team A won, 0 when team B won, <NA> when not yet played.
TARGET_COLUMN = "team_a_won"

#: Not model inputs: the provider-Elo baseline and the ranking snapshot used.
#: ``provider_elo_prob`` is the win probability for A implied by the provider's
#: Elo as published before the match -- the bar the model has to beat.
BASELINE_COLUMNS: tuple[str, ...] = ("provider_elo_prob", "ranking_published_at")

#: Identifying columns carried alongside the features for joins and debugging.
KEY_COLUMNS: tuple[str, ...] = (
    "match_id",
    "played_at",
    "tournament",
    "tier",
    "round",
    "team_a_player_1",
    "team_a_player_2",
    "team_b_player_1",
    "team_b_player_2",
)

#: Rest days assumed for a player who has not been seen before.
DEFAULT_REST_DAYS = 14.0
#: Rest days are capped so that a long absence does not dominate the feature.
MAX_REST_DAYS = 60.0
#: Win rate assumed for a player with no match history.
DEFAULT_FORM = 0.5


def expected_score(elo_a: float, elo_b: float) -> float:
    """Standard Elo expectation that team A beats team B."""
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


class MatchState:
    """Rolling per-player and per-pair state, advanced one match at a time.

    The state is intentionally cheap to rebuild: replaying the full match history
    takes milliseconds, so training and inference both start from the same
    deterministic state instead of persisting mutable ratings somewhere.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or get_pipeline_config().features
        self.elo: dict[str, float] = defaultdict(lambda: self.config.elo_initial)
        self.recent: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=max(1, self.config.form_window))
        )
        self.last_played: dict[str, pd.Timestamp] = {}
        self.pair_matches: dict[tuple[str, str], int] = defaultdict(int)

    # -- readers ---------------------------------------------------------- #
    @staticmethod
    def pair_key(players: Iterable[str]) -> tuple[str, str]:
        """Order-independent key for a pair of players."""
        first, second = sorted(str(p) for p in players)
        return first, second

    def team_elo(self, players: Sequence[str]) -> float:
        """Team rating: the mean of its two players' ratings."""
        return sum(self.elo[p] for p in players) / len(players)

    def team_form(self, players: Sequence[str]) -> float:
        """Mean win rate of the team's players over the rolling form window."""
        rates = [
            sum(self.recent[p]) / len(self.recent[p]) if self.recent[p] else DEFAULT_FORM
            for p in players
        ]
        return sum(rates) / len(rates)

    def team_rest_days(self, players: Sequence[str], played_at: pd.Timestamp) -> float:
        """Days since the team's *freshest* player last played, capped."""
        gaps = []
        for player in players:
            previous = self.last_played.get(player)
            if previous is None:
                gaps.append(DEFAULT_REST_DAYS)
            else:
                gaps.append((played_at - previous).total_seconds() / 86400.0)
        return float(min(max(min(gaps), 0.0), MAX_REST_DAYS))

    def partnership_matches(self, players: Sequence[str]) -> int:
        """How many matches this exact pair has already played together."""
        return self.pair_matches[self.pair_key(players)]

    def tier_level(self, tier: object) -> int:
        """Ordinal encoding of the tour tier; unknown tiers map to 0."""
        key = str(tier).strip().lower() if tier is not None and pd.notna(tier) else ""
        return int(self.config.tier_levels.get(key, 0))

    # -- the feature row -------------------------------------------------- #
    def features_for(
        self,
        team_a: Sequence[str],
        team_b: Sequence[str],
        played_at: pd.Timestamp,
        tier: object,
        rank_points_a: float | None,
        rank_points_b: float | None,
    ) -> dict[str, float]:
        """Build the feature row for one match from the *current* state.

        Must be called before :meth:`update` for that same match.
        """
        rank_a = 0.0 if rank_points_a is None or pd.isna(rank_points_a) else float(rank_points_a)
        rank_b = 0.0 if rank_points_b is None or pd.isna(rank_points_b) else float(rank_points_b)
        return {
            "elo_diff": self.team_elo(team_a) - self.team_elo(team_b),
            "rank_points_diff": rank_a - rank_b,
            "form_diff": self.team_form(team_a) - self.team_form(team_b),
            "rest_days_diff": self.team_rest_days(team_a, played_at)
            - self.team_rest_days(team_b, played_at),
            "partnership_diff": float(
                self.partnership_matches(team_a) - self.partnership_matches(team_b)
            ),
            "tier_level": float(self.tier_level(tier)),
        }

    # -- writer ----------------------------------------------------------- #
    def update(
        self,
        team_a: Sequence[str],
        team_b: Sequence[str],
        played_at: pd.Timestamp,
        team_a_won: int,
    ) -> None:
        """Fold a finished match into the state (Elo, form, rest, partnerships)."""
        elo_a, elo_b = self.team_elo(team_a), self.team_elo(team_b)
        delta = self.config.elo_k_factor * (team_a_won - expected_score(elo_a, elo_b))

        for player in team_a:
            self.elo[player] += delta
            self.recent[player].append(team_a_won)
            self.last_played[player] = played_at
        for player in team_b:
            self.elo[player] -= delta
            self.recent[player].append(1 - team_a_won)
            self.last_played[player] = played_at

        self.pair_matches[self.pair_key(team_a)] += 1
        self.pair_matches[self.pair_key(team_b)] += 1


_SIDES: tuple[tuple[str, tuple[str, str]], ...] = (
    ("team_a", ("team_a_player_1", "team_a_player_2")),
    ("team_b", ("team_b_player_1", "team_b_player_2")),
)


def attach_rankings(matches: pd.DataFrame, rankings: pd.DataFrame) -> pd.DataFrame:
    """As-of join of ranking snapshots onto matches, strictly before the start.

    For each of the four players the latest snapshot with
    ``published_at < played_at`` is used (``allow_exact_matches=False``). A
    snapshot published at or after the scheduled start -- for instance one
    labelled with the current tournament week but published once that week is
    over -- is never visible to the match.

    Overwrites ``team_{a,b}_rank_points`` (sum of the two players; NaN when
    neither player is ranked yet) and adds :data:`BASELINE_COLUMNS`.
    """
    from padel_predictor.common.schemas import normalise_raw_rankings

    out = matches.copy()
    ranks = normalise_raw_rankings(rankings)
    ranks = ranks.sort_values("published_at", kind="stable")
    played_at = out["played_at"].astype("datetime64[ns, UTC]")

    per_side: dict[str, dict[str, pd.Series]] = {}
    published: list[pd.Series] = []
    for side, columns in _SIDES:
        points, elos = [], []
        for column in columns:
            left = pd.DataFrame(
                {"_row": range(len(out)), "played_at": played_at.to_numpy(), "player": out[column]}
            )
            left["player"] = left["player"].astype("string")
            left = left.sort_values("played_at", kind="stable")
            joined = pd.merge_asof(
                left,
                ranks[["player", "published_at", "rank_points", "provider_elo"]],
                left_on="played_at",
                right_on="published_at",
                by="player",
                allow_exact_matches=False,
                direction="backward",
            ).sort_values("_row")
            points.append(joined["rank_points"].reset_index(drop=True))
            elos.append(joined["provider_elo"].reset_index(drop=True))
            published.append(joined["published_at"].reset_index(drop=True))
        per_side[side] = {
            "points": pd.concat(points, axis=1).sum(axis=1, min_count=1),
            # The provider's pair rating needs both players; otherwise no baseline.
            "elo": pd.concat(elos, axis=1).mean(axis=1, skipna=False),
        }

    index = out.index
    out["team_a_rank_points"] = per_side["team_a"]["points"].set_axis(index).astype("Float64")
    out["team_b_rank_points"] = per_side["team_b"]["points"].set_axis(index).astype("Float64")
    elo_a = per_side["team_a"]["elo"].set_axis(index)
    elo_b = per_side["team_b"]["elo"].set_axis(index)
    out["provider_elo_prob"] = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    out["ranking_published_at"] = pd.concat(published, axis=1).max(axis=1).set_axis(index)
    return out


def build_feature_frame(
    raw_matches: pd.DataFrame,
    config: FeatureConfig | None = None,
    rankings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn raw matches into model-ready feature rows.

    Args:
        raw_matches: frame satisfying the raw-match contract. May mix finished
            matches (``winner`` set) and upcoming ones (``winner`` null).
        config: feature parameters; defaults to ``config/settings.yaml``.
        rankings: ranking snapshots (:data:`~padel_predictor.common.schemas.RAW_RANKING_COLUMNS`).
            When given -- as every pipeline does -- ranking points and the
            provider-Elo baseline come from the as-of join in
            :func:`attach_rankings`, and the rank columns of ``raw_matches`` are
            ignored. An empty frame means "no snapshot published yet".

    Returns:
        One row per input match, with :data:`KEY_COLUMNS`, :data:`FEATURE_COLUMNS`
        and :data:`TARGET_COLUMN`. The target is ``<NA>`` for upcoming matches,
        whose feature rows are still fully populated -- that is exactly what the
        inference pipeline consumes.
    """
    matches = normalise_raw_matches(raw_matches)
    if matches.empty:
        return pd.DataFrame(
            columns=[*KEY_COLUMNS, *FEATURE_COLUMNS, *BASELINE_COLUMNS, TARGET_COLUMN]
        )
    if rankings is not None:
        matches = attach_rankings(matches, rankings)
    else:
        matches = matches.assign(provider_elo_prob=float("nan"), ranking_published_at=pd.NaT)

    state = MatchState(config)
    rows: list[dict[str, object]] = []

    for match in matches.itertuples(index=False):
        team_a = (match.team_a_player_1, match.team_a_player_2)
        team_b = (match.team_b_player_1, match.team_b_player_2)

        # 1. Features first: the state holds only strictly earlier matches.
        row: dict[str, object] = {
            key: getattr(match, key) for key in (*KEY_COLUMNS, *BASELINE_COLUMNS)
        }
        row.update(
            state.features_for(
                team_a=team_a,
                team_b=team_b,
                played_at=match.played_at,
                tier=match.tier,
                rank_points_a=match.team_a_rank_points,
                rank_points_b=match.team_b_rank_points,
            )
        )

        # 2. Then the label, and only then the state update.
        if pd.isna(match.winner):
            row[TARGET_COLUMN] = pd.NA
        else:
            team_a_won = int(match.winner == "A")
            row[TARGET_COLUMN] = team_a_won
            state.update(team_a, team_b, match.played_at, team_a_won)

        rows.append(row)

    features = pd.DataFrame(rows)
    features[list(FEATURE_COLUMNS)] = features[list(FEATURE_COLUMNS)].astype("float64")
    features["provider_elo_prob"] = features["provider_elo_prob"].astype("float64")
    features["ranking_published_at"] = pd.to_datetime(features["ranking_published_at"], utc=True)
    features[TARGET_COLUMN] = features[TARGET_COLUMN].astype("Int64")
    return features


def labelled(features: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known outcome -- the training pipeline's input."""
    return features.loc[features[TARGET_COLUMN].notna()].reset_index(drop=True)


def unlabelled(features: pd.DataFrame) -> pd.DataFrame:
    """Rows without an outcome yet -- the inference pipeline's input."""
    return features.loc[features[TARGET_COLUMN].isna()].reset_index(drop=True)
