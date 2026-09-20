"""Feature definitions -- the single source of truth for the whole system.

Both the training pipeline and the inference pipeline import
:func:`build_feature_frame` from this module. There is no second implementation
of "how a match becomes a feature row" anywhere in the repository, which is what
keeps training and serving from drifting apart.

**No leakage by construction.** Features are derived from rolling state (Elo,
recent form, rest, partnership history) replayed in chronological order. For
every match the feature row is computed *before* that match's result is folded
into the state, so a row only ever sees matches that finished earlier.
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


def build_feature_frame(
    raw_matches: pd.DataFrame,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Turn raw matches into model-ready feature rows.

    Args:
        raw_matches: frame satisfying the raw-match contract. May mix finished
            matches (``winner`` set) and upcoming ones (``winner`` null).
        config: feature parameters; defaults to ``config/settings.yaml``.

    Returns:
        One row per input match, with :data:`KEY_COLUMNS`, :data:`FEATURE_COLUMNS`
        and :data:`TARGET_COLUMN`. The target is ``<NA>`` for upcoming matches,
        whose feature rows are still fully populated -- that is exactly what the
        inference pipeline consumes.
    """
    matches = normalise_raw_matches(raw_matches)
    if matches.empty:
        return pd.DataFrame(columns=[*KEY_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN])

    state = MatchState(config)
    rows: list[dict[str, object]] = []

    for match in matches.itertuples(index=False):
        team_a = (match.team_a_player_1, match.team_a_player_2)
        team_b = (match.team_b_player_1, match.team_b_player_2)

        # 1. Features first: the state holds only strictly earlier matches.
        row: dict[str, object] = {key: getattr(match, key) for key in KEY_COLUMNS}
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
    features[TARGET_COLUMN] = features[TARGET_COLUMN].astype("Int64")
    return features


def labelled(features: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known outcome -- the training pipeline's input."""
    return features.loc[features[TARGET_COLUMN].notna()].reset_index(drop=True)


def unlabelled(features: pd.DataFrame) -> pd.DataFrame:
    """Rows without an outcome yet -- the inference pipeline's input."""
    return features.loc[features[TARGET_COLUMN].isna()].reset_index(drop=True)
