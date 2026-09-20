"""Command line entrypoint.

One command per FTI pipeline, so a scheduled workflow, a Docker container and a
developer all start the same code the same way::

    padel feature-pipeline
    padel backfill --start 2026-01-01 --end 2026-09-01
    padel training-pipeline
    padel inference-pipeline
    padel matchup --team-a "Player 01,Player 02" --team-b "Player 03,Player 04"
    padel serve
"""

from __future__ import annotations

import datetime as dt

import typer

from padel_predictor.common.logging import get_logger, setup_logging

app = typer.Typer(
    help="Padel Predictor: feature, training and inference pipelines.",
    no_args_is_help=True,
    add_completion=False,
)
logger = get_logger(__name__)


def _parse_team(value: str) -> tuple[str, str]:
    players = [p.strip() for p in value.split(",") if p.strip()]
    if len(players) != 2:
        raise typer.BadParameter(f"expected two comma-separated players, got {value!r}")
    return players[0], players[1]


@app.callback()
def main() -> None:
    """Configure logging before any subcommand runs."""
    setup_logging()


@app.command("feature-pipeline")
def feature_pipeline(
    lookback_days: int | None = typer.Option(
        None, help="Days of history to fetch. Defaults to config/settings.yaml."
    ),
    horizon_days: int | None = typer.Option(
        None, help="Days of upcoming fixtures to fetch. Defaults to config/settings.yaml."
    ),
) -> None:
    """Ingest recent matches and rebuild the feature store."""
    from padel_predictor.features import pipeline

    result = pipeline.run(lookback_days=lookback_days, horizon_days=horizon_days)
    typer.echo(result.summary())


@app.command("backfill")
def backfill(
    start: str = typer.Option(..., help="First day to load, YYYY-MM-DD."),
    end: str = typer.Option(..., help="Last day to load, YYYY-MM-DD."),
    chunk_days: int = typer.Option(30, help="Days per request to the live source."),
) -> None:
    """Load a historical date range into the feature store."""
    from padel_predictor.features import pipeline

    result = pipeline.backfill(
        start=dt.date.fromisoformat(start),
        end=dt.date.fromisoformat(end),
        chunk_days=chunk_days,
    )
    typer.echo(result.summary())


@app.command("training-pipeline")
def training_pipeline(
    register: bool = typer.Option(
        True, "--register/--no-register", help="Log to MLflow and promote if better."
    ),
) -> None:
    """Train a model, evaluate it against the champion, promote it if better."""
    from padel_predictor.training import pipeline

    result = pipeline.run(register=register)
    typer.echo(result.summary())


@app.command("inference-pipeline")
def inference_pipeline() -> None:
    """Predict every upcoming match and store the predictions."""
    from padel_predictor.inference import pipeline

    result = pipeline.run()
    typer.echo(result.summary())


@app.command("matchup")
def matchup(
    team_a: str = typer.Option(..., help='Two players, comma separated: "Name 1,Name 2".'),
    team_b: str = typer.Option(..., help='Two players, comma separated: "Name 3,Name 4".'),
    tier: str = typer.Option("p1", help="Tour tier: major, p1, p2 or challenger."),
) -> None:
    """Predict a single, possibly hypothetical, matchup."""
    from padel_predictor.inference.predict import Predictor, load_history

    predictor = Predictor.load()
    prediction = predictor.predict_matchup(
        history=load_history(),
        team_a=_parse_team(team_a),
        team_b=_parse_team(team_b),
        tier=tier,
    )
    typer.echo(
        f"{prediction.team_a} vs {prediction.team_b}\n"
        f"  P(team A wins) = {prediction.probability_team_a:.3f}\n"
        f"  predicted winner: team {prediction.predicted_winner}\n"
        f"  model: {prediction.model_name} v{prediction.model_version}"
    )


@app.command("serve")
def serve() -> None:
    """Run the prediction API."""
    from padel_predictor.inference.api import main as serve_api

    serve_api()


if __name__ == "__main__":  # pragma: no cover
    app()
