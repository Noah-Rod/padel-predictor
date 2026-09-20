"""Streamlit front end.

Talks to the prediction API when one is reachable and falls back to loading the
model in-process otherwise, so the same file works in Docker Compose (two
services) and on a single-container host such as a Hugging Face Space.
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

from padel_predictor.common.config import get_pipeline_config, get_settings
from padel_predictor.inference.predict import (
    ModelNotAvailableError,
    Predictor,
    known_players,
    load_history,
)

TIERS = ("major", "p1", "p2", "challenger")

st.set_page_config(page_title="Padel Predictor", page_icon="🎾", layout="wide")


@st.cache_resource(show_spinner="Loading model and match history…")
def _local_predictor() -> tuple[Predictor, pd.DataFrame]:
    settings = get_settings()
    return Predictor.load(settings, get_pipeline_config()), load_history(settings)


def _api_url() -> str | None:
    url = os.getenv("PADEL_API_URL", "").strip()
    if not url:
        return None
    try:
        response = httpx.get(f"{url.rstrip('/')}/health", timeout=3.0)
        if response.status_code == 200 and response.json().get("model_loaded"):
            return url.rstrip("/")
    except httpx.HTTPError:
        return None
    return None


def main() -> None:
    st.title("🎾 Padel Predictor")
    st.caption(
        "Predicts the winner of a professional padel match from Elo, recent form, "
        "rest, partnership history and ranking points."
    )

    api = _api_url()
    try:
        predictor, history = _local_predictor()
    except (ModelNotAvailableError, RuntimeError) as exc:
        st.error(f"The system is not ready yet: {exc}")
        st.info(
            "Run the pipelines first:\n\n"
            "```bash\n"
            "padel backfill --start 2026-01-01 --end 2026-09-01\n"
            "padel training-pipeline\n"
            "```"
        )
        return

    left, right = st.columns([2, 1])
    with right:
        st.metric("Model version", predictor.model_version)
        st.metric("Matches in history", f"{len(history):,}")
        st.caption(f"Serving mode: {'API' if api else 'in-process'}")

    players = known_players(history)
    with left:
        st.subheader("Predict a matchup")
        team_a = st.multiselect("Team A", players, default=players[:2], max_selections=2)
        team_b = st.multiselect("Team B", players, default=players[2:4], max_selections=2)
        tier = st.selectbox("Tour tier", TIERS, index=TIERS.index("p1"))

        if st.button("Predict", type="primary", disabled=len(team_a) != 2 or len(team_b) != 2):
            prediction = predictor.predict_matchup(
                history=history,
                team_a=(team_a[0], team_a[1]),
                team_b=(team_b[0], team_b[1]),
                tier=tier,
            )
            probability = prediction.probability_team_a
            winner = "Team A" if prediction.predicted_winner == "A" else "Team B"
            st.success(f"**{winner}** is favoured — P(Team A wins) = {probability:.1%}")
            st.progress(probability)

    st.divider()
    st.subheader("Upcoming matches")
    upcoming = predictor.predict_upcoming(history)
    if upcoming.empty:
        st.info("No scheduled matches inside the prediction horizon.")
        return

    table = pd.DataFrame(
        {
            "Date": upcoming["played_at"].dt.strftime("%Y-%m-%d %H:%M"),
            "Tournament": upcoming["tournament"],
            "Round": upcoming["round"],
            "Team A": upcoming["team_a_player_1"] + " / " + upcoming["team_a_player_2"],
            "Team B": upcoming["team_b_player_1"] + " / " + upcoming["team_b_player_2"],
            "P(Team A)": upcoming["probability_team_a"],
        }
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "P(Team A)": st.column_config.ProgressColumn(
                "P(Team A)", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )


if __name__ == "__main__":
    main()
