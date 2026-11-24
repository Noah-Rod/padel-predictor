"""Streamlit front end.

Three views on one page: the upcoming matches with the champion's prediction,
a matchup builder for any four players, and the track record of the stored
predictions. The model and the store are loaded in this process, so the same
file runs next to the API in Docker Compose and alone on a single-container
host. Everything it shows is computed in
:mod:`padel_predictor.inference.dashboard`; this file only lays it out.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from padel_predictor.common.config import get_pipeline_config, get_settings
from padel_predictor.inference.dashboard import (
    Dashboard,
    load_dashboard,
    player_table,
    team_comparison,
    weekly_performance,
)
from padel_predictor.inference.predict import ModelNotAvailableError

VIEWS = ("Upcoming matches", "Matchup builder", "Track record")
TIER_LABELS = {
    "major": "Major",
    "finals": "Finals",
    "p1": "P1",
    "p2": "P2",
    "fip_platinum": "FIP Platinum",
    "fip_gold": "FIP Gold",
    "challenger": "Challenger",
}
#: The store is read again after this long, so a new champion and new fixtures
#: show up without a restart.
REFRESH_AFTER = "1h"
MODEL_COLOUR = "#2a78d6"
BASELINE_COLOUR = "#8a8984"

st.set_page_config(page_title="Padel Predictor", page_icon="🎾", layout="wide")


@st.cache_resource(ttl=REFRESH_AFTER, show_spinner="Loading the model and the match history…")
def _dashboard() -> Dashboard:
    return load_dashboard(get_settings(), get_pipeline_config())


@st.cache_data(show_spinner=False)
def _predict(
    loaded_at: str,
    team_a: tuple[str, str],
    team_b: tuple[str, str],
    tier: str,
    played_at: pd.Timestamp | None,
) -> dict:
    """Predict one matchup; cached per data load, so a repeat costs nothing."""
    dash = _dashboard()
    row = dash.predictor.predict_matchup_row(
        dash.history, team_a, team_b, tier, played_at=played_at
    )
    return {
        "probability": float(row["probability_team_a"]),
        "played_at": row["played_at"],
        "rank_points_diff": float(row["rank_points_diff"]),
    }


def _tiers() -> list[str]:
    levels = get_pipeline_config().features.tier_levels
    return sorted(levels, key=lambda tier: (-levels[tier], tier))


def _tier_label(tier: str) -> str:
    return TIER_LABELS.get(tier, tier)


def _pair(first: str, second: str) -> str:
    return f"{first} / {second}"


def _set_matchup(match: pd.Series) -> None:
    """Put a scheduled match into the builder.

    Its start time is kept as well: rest days and the ranking snapshot depend
    on the date, so the unchanged matchup gives the scheduled prediction, and
    swapping a player answers "what if they played this match instead".
    """
    tiers = _tiers()
    st.session_state["team_a"] = [match["team_a_player_1"], match["team_a_player_2"]]
    st.session_state["team_b"] = [match["team_b_player_1"], match["team_b_player_2"]]
    st.session_state["tier"] = match["tier"] if match["tier"] in tiers else tiers[0]
    st.session_state["played_at"] = match["played_at"]
    st.session_state["slot"] = f"{match['tournament']} · {match['round']}"


def _load_upcoming(match: pd.Series) -> None:
    _set_matchup(match)
    st.session_state["view"] = "Matchup builder"


def _clear_slot() -> None:
    st.session_state["played_at"] = None
    st.session_state["slot"] = None


def _init_builder(dash: Dashboard) -> None:
    """First visit: start from the next upcoming match, else the four top-rated players."""
    if "team_a" in st.session_state:
        return
    tiers = _tiers()
    if not dash.upcoming.empty:
        _set_matchup(dash.upcoming.iloc[0])
        return
    _clear_slot()
    top = player_table(dash.history, dash.state).nlargest(4, "elo")["player"].tolist()
    st.session_state["team_a"] = top[:2]
    st.session_state["team_b"] = top[2:4]
    st.session_state["tier"] = "p1" if "p1" in tiers else tiers[0]


# --------------------------------------------------------------------------- #
# Shared pieces
# --------------------------------------------------------------------------- #
def _probability_bar(probability_a: float) -> None:
    """A two-sided bar: team A from the left, team B from the right."""
    share = max(0.0, min(1.0, probability_a)) * 100
    st.markdown(
        f"""
        <div style="display:flex;gap:2px;height:14px;margin:0.25rem 0 1rem 0"
             role="img" aria-label="Team A {share:.0f} percent, team B {100 - share:.0f} percent">
          <div style="width:{share:.2f}%;background:{MODEL_COLOUR};border-radius:4px 0 0 4px"></div>
          <div style="flex:1;background:{BASELINE_COLOUR};border-radius:0 4px 4px 0"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _matchup_card(
    dash: Dashboard,
    team_a: tuple[str, str],
    team_b: tuple[str, str],
    probability_a: float,
    played_at: pd.Timestamp,
    rank_points_diff: float,
) -> None:
    """The prediction for one matchup and the numbers the model was given."""
    favourite = _pair(*team_a) if probability_a >= 0.5 else _pair(*team_b)
    confidence = max(probability_a, 1 - probability_a)
    st.markdown(f"**{favourite}** are favoured, with a {confidence:.0%} chance to win.")

    left, right = st.columns(2)
    left.metric(f"Team A · {_pair(*team_a)}", f"{probability_a:.1%}")
    right.metric(f"Team B · {_pair(*team_b)}", f"{1 - probability_a:.1%}")
    _probability_bar(probability_a)

    comparison = team_comparison(dash.state, team_a, team_b, played_at)
    comparison["difference"] = comparison["team_a"] - comparison["team_b"]
    st.dataframe(
        comparison.rename(
            columns={
                "measure": "What the model sees",
                "team_a": "Team A",
                "team_b": "Team B",
                "difference": "A minus B",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Team A": st.column_config.NumberColumn(format="%.2f"),
            "Team B": st.column_config.NumberColumn(format="%.2f"),
            "A minus B": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    st.caption(
        f"Official ranking points, A minus B: {rank_points_diff:+,.0f}. "
        "The model reads the differences, plus the tier of the event."
    )


# --------------------------------------------------------------------------- #
# Views
# --------------------------------------------------------------------------- #
def _upcoming_view(dash: Dashboard) -> None:
    upcoming = dash.upcoming
    if upcoming.empty:
        st.info(
            "No scheduled matches in the next "
            f"{dash.predictor.config.inference.horizon_days} days. "
            "The matchup builder predicts any four players in the meantime."
        )
        return

    tournaments = sorted(upcoming["tournament"].dropna().unique())
    chosen = st.selectbox("Tournament", ["All tournaments", *tournaments])
    if chosen != "All tournaments":
        upcoming = upcoming.loc[upcoming["tournament"] == chosen]
    upcoming = upcoming.reset_index(drop=True)

    probability = upcoming["probability_team_a"]
    team_a = upcoming["team_a_player_1"] + " / " + upcoming["team_a_player_2"]
    team_b = upcoming["team_b_player_1"] + " / " + upcoming["team_b_player_2"]
    table = pd.DataFrame(
        {
            "Date": upcoming["played_at"].dt.strftime("%a %d.%m. %H:%M"),
            "Tournament": upcoming["tournament"],
            "Round": upcoming["round"],
            "Team A": team_a,
            "Team B": team_b,
            "Favourite": team_a.where(probability >= 0.5, team_b),
            "Chance": probability.where(probability >= 0.5, 1 - probability),
        }
    )
    st.caption(f"{len(table)} matches, times in UTC. Select a row to see the prediction.")
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="upcoming_table",
        column_config={
            "Chance": st.column_config.ProgressColumn(
                "Chance", min_value=0.0, max_value=1.0, format="percent"
            )
        },
    )

    rows = event.selection.rows
    if not rows or rows[0] >= len(upcoming):
        return
    match = upcoming.iloc[rows[0]]
    with st.container(border=True):
        st.subheader(f"{match['tournament']} · {match['round']}")
        _matchup_card(
            dash,
            (match["team_a_player_1"], match["team_a_player_2"]),
            (match["team_b_player_1"], match["team_b_player_2"]),
            float(match["probability_team_a"]),
            match["played_at"],
            float(match["rank_points_diff"]),
        )
        st.button(
            "Change the players in the matchup builder",
            on_click=_load_upcoming,
            args=(match,),
        )


def _pick_upcoming(dash: Dashboard) -> None:
    index = st.session_state.get("upcoming_pick")
    if index is not None:
        _load_upcoming(dash.upcoming.iloc[index])
        st.session_state["upcoming_pick"] = None


def _builder_view(dash: Dashboard) -> None:
    players = player_table(dash.history, dash.state)
    names = players["player"].tolist()

    if not dash.upcoming.empty:

        def label(index: int) -> str:
            match = dash.upcoming.iloc[index]
            return (
                f"{match['played_at']:%d.%m.} · "
                f"{_pair(match['team_a_player_1'], match['team_a_player_2'])} vs "
                f"{_pair(match['team_b_player_1'], match['team_b_player_2'])}"
            )

        st.selectbox(
            "Start from an upcoming match",
            range(len(dash.upcoming)),
            index=None,
            format_func=label,
            placeholder="Choose a scheduled match, then change any player",
            key="upcoming_pick",
            on_change=_pick_upcoming,
            args=(dash,),
        )

    left, right, side = st.columns([2, 2, 1])
    team_a = left.multiselect("Team A", names, max_selections=2, key="team_a")
    team_b = right.multiselect("Team B", names, max_selections=2, key="team_b")
    tier = side.selectbox("Event tier", _tiers(), format_func=_tier_label, key="tier")

    if len(team_a) != 2 or len(team_b) != 2:
        st.info("Choose two players for each team.")
        return
    both = sorted(set(team_a) & set(team_b))
    if both:
        st.warning(f"{', '.join(both)} cannot play on both teams.")
        return

    played_at = st.session_state.get("played_at")
    prediction = _predict(dash.loaded_at.isoformat(), tuple(team_a), tuple(team_b), tier, played_at)
    with st.container(border=True):
        if played_at is None:
            st.caption("Predicted as if the match were played tomorrow.")
        else:
            note, reset = st.columns([4, 1])
            note.caption(
                f"Predicted for the slot of {st.session_state.get('slot')}, "
                f"{played_at:%a %d.%m.%Y %H:%M} UTC."
            )
            reset.button("Use tomorrow instead", on_click=_clear_slot)
        _matchup_card(
            dash,
            tuple(team_a),
            tuple(team_b),
            prediction["probability"],
            prediction["played_at"],
            prediction["rank_points_diff"],
        )

    st.subheader("The four players")
    chosen = players.set_index("player").loc[[*team_a, *team_b]].reset_index()
    chosen.insert(0, "team", ["A", "A", "B", "B"])
    st.dataframe(
        chosen.rename(
            columns={
                "team": "Team",
                "player": "Player",
                "elo": "In-house Elo",
                "form": "Recent form",
                "matches": "Matches",
                "last_played": "Last match",
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "In-house Elo": st.column_config.NumberColumn(format="%.0f"),
            "Recent form": st.column_config.NumberColumn(format="percent"),
            "Last match": st.column_config.DatetimeColumn(format="DD.MM.YYYY"),
        },
    )


def _track_record_view(dash: Dashboard) -> None:
    scored = dash.scored
    if scored.empty:
        st.info(
            "No stored prediction has a result yet. The daily inference run stores "
            "its predictions; they are scored here once the matches are played."
        )
        return

    rated = scored.loc[scored["provider_elo_prob"].notna()]
    first, second, third, fourth = st.columns(4)
    first.metric("Matches scored", f"{len(scored):,}")
    second.metric("Favourite won", f"{scored['correct'].mean():.1%}")
    third.metric("Log loss", f"{scored['log_loss'].mean():.3f}")
    if rated.empty:
        fourth.metric("Provider Elo log loss", "n/a")
    else:
        gap = rated["log_loss"].mean() - rated["provider_log_loss"].mean()
        fourth.metric(
            f"Provider Elo log loss ({len(rated):,} rated)",
            f"{rated['provider_log_loss'].mean():.3f}",
            delta=f"{gap:+.3f} model against provider",
            delta_color="inverse",
        )
    st.caption(
        "Each finished match is scored on the last prediction stored before it was "
        "played. Log loss: lower is better; 0.693 is a coin flip."
    )

    weekly = weekly_performance(scored)
    weekly["label"] = weekly["week"].dt.strftime("%d.%m.")
    week_axis = alt.X("label:O", title="Week starting", sort=None, axis=alt.Axis(labelAngle=0))
    st.subheader("Log loss per week, on the matches the provider rates")
    lines = weekly.melt(
        id_vars=["week", "label", "rated"],
        value_vars=["log_loss_rated", "provider_log_loss"],
        var_name="series",
        value_name="value",
    ).dropna(subset=["value"])
    lines["series"] = lines["series"].map(
        {"log_loss_rated": "Model", "provider_log_loss": "Provider Elo"}
    )
    if lines.empty:
        st.caption("The provider has no rating for any scored match yet.")
    else:
        st.altair_chart(
            alt.Chart(lines)
            .mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2)
            .encode(
                x=week_axis,
                y=alt.Y("value:Q", title="Log loss", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Model", "Provider Elo"],
                        range=[MODEL_COLOUR, BASELINE_COLOUR],
                    ),
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("week:T", title="Week starting"),
                    alt.Tooltip("series:N", title="Scored"),
                    alt.Tooltip("value:Q", title="Log loss", format=".3f"),
                    alt.Tooltip("rated:Q", title="Rated matches"),
                ],
            ),
            width="stretch",
        )

    st.subheader("Favourite won, per week")
    st.altair_chart(
        alt.Chart(weekly)
        .mark_bar(color=MODEL_COLOUR, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=24)
        .encode(
            x=week_axis,
            y=alt.Y(
                "accuracy:Q",
                title="Share of matches",
                axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1]),
            ),
            tooltip=[
                alt.Tooltip("week:T", title="Week starting"),
                alt.Tooltip("accuracy:Q", title="Favourite won", format=".1%"),
                alt.Tooltip("matches:Q", title="Matches"),
            ],
        ),
        width="stretch",
    )

    with st.expander("The weekly numbers as a table"):
        st.dataframe(
            weekly.drop(columns="label").rename(
                columns={
                    "week": "Week starting",
                    "matches": "Matches",
                    "accuracy": "Favourite won",
                    "log_loss": "Log loss",
                    "rated": "Rated matches",
                    "log_loss_rated": "Log loss (rated)",
                    "provider_log_loss": "Provider Elo log loss",
                }
            ),
            hide_index=True,
            width="stretch",
            column_config={
                "Week starting": st.column_config.DateColumn(format="DD.MM.YYYY"),
                "Favourite won": st.column_config.NumberColumn(format="percent"),
                "Log loss": st.column_config.NumberColumn(format="%.3f"),
                "Log loss (rated)": st.column_config.NumberColumn(format="%.3f"),
                "Provider Elo log loss": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    st.subheader("Latest results")
    recent = scored.sort_values("played_at", ascending=False).head(50)
    team_a = recent["team_a_player_1"] + " / " + recent["team_a_player_2"]
    team_b = recent["team_b_player_1"] + " / " + recent["team_b_player_2"]
    a_favoured = recent["probability_team_a"] >= 0.5
    a_won = recent["team_a_won"] == 1
    st.dataframe(
        pd.DataFrame(
            {
                "Date": recent["played_at"].dt.strftime("%d.%m.%Y"),
                "Tournament": recent["tournament"],
                "Round": recent["round"],
                "Favourite": team_a.where(a_favoured, team_b),
                "Chance": recent["probability_team_a"].where(
                    a_favoured, 1 - recent["probability_team_a"]
                ),
                "Winner": team_a.where(a_won, team_b),
                "Result": recent["correct"].map({True: "✓ favourite won", False: "✗ upset"}),
                "Model": recent["model_version"],
            }
        ),
        hide_index=True,
        width="stretch",
        column_config={
            "Chance": st.column_config.ProgressColumn(
                "Chance", min_value=0.0, max_value=1.0, format="percent"
            )
        },
    )


def main() -> None:
    st.title("🎾 Padel Predictor")
    st.caption(
        "Who wins a professional padel match, from Elo, recent form, rest, "
        "partnership history and ranking points."
    )

    try:
        dash = _dashboard()
    except (ModelNotAvailableError, RuntimeError) as exc:
        st.error(f"The system is not ready yet: {exc}")
        st.info(
            "Run the pipelines first:\n\n"
            "```bash\n"
            "padel backfill --start 2025-09-01 --end 2026-09-01\n"
            "padel training-pipeline\n"
            "```"
        )
        return
    _init_builder(dash)

    finished = int(dash.history["winner"].notna().sum())
    first, second, third, fourth = st.columns(4)
    first.metric("Model version", dash.predictor.model_version)
    second.metric("Finished matches", f"{finished:,}")
    third.metric("Upcoming matches", f"{len(dash.upcoming):,}")
    fourth.metric("Data loaded (UTC)", f"{dash.loaded_at:%d.%m. %H:%M}")
    if fourth.button("Reload now", help="Read the store and the champion again"):
        _dashboard.clear()
        _predict.clear()
        st.rerun()

    st.session_state.setdefault("view", VIEWS[0])
    view = st.segmented_control("View", VIEWS, key="view", label_visibility="collapsed")
    if view == "Matchup builder":
        _builder_view(dash)
    elif view == "Track record":
        _track_record_view(dash)
    else:
        _upcoming_view(dash)


if __name__ == "__main__":
    main()
