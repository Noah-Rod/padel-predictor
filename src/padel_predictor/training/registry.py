"""MLflow tracking and model registry.

Every MLflow call in the project goes through this module. ``mlflow`` is
imported lazily inside the functions so that importing the package -- which the
unit tests and the feature pipeline do -- does not pay for it.

Model versions are addressed by *alias* (``@champion``) rather than by the
deprecated stage API, so the inference pipeline asks for "the best model" and
never hard-codes a version number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from padel_predictor.common.config import Settings, get_settings
from padel_predictor.common.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelHandle:
    """A loaded model plus the provenance the UI and API report."""

    model: Any
    name: str
    version: str
    alias: str
    run_id: str | None = None

    @property
    def uri(self) -> str:
        return f"models:/{self.name}/{self.version}"


def configure(settings: Settings | None = None) -> Any:
    """Point MLflow at the configured tracking server and experiment."""
    import mlflow

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    return mlflow


def get_client(settings: Settings | None = None) -> Any:
    """An ``MlflowClient`` bound to the configured tracking URI."""
    from mlflow.tracking import MlflowClient

    settings = settings or get_settings()
    return MlflowClient(tracking_uri=settings.mlflow_tracking_uri)


def champion_version(settings: Settings | None = None) -> Any | None:
    """The registered version currently carrying the champion alias, if any."""
    settings = settings or get_settings()
    try:
        return get_client(settings).get_model_version_by_alias(
            settings.registered_model_name, settings.model_alias
        )
    except Exception as exc:  # noqa: BLE001 -- a missing model is the normal first-run case
        logger.info(
            "no %r alias on model %r yet (%s)",
            settings.model_alias,
            settings.registered_model_name,
            type(exc).__name__,
        )
        return None


def load_champion(settings: Settings | None = None) -> ModelHandle | None:
    """Load the current champion model, or ``None`` when nothing is registered."""
    import mlflow.sklearn

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    version = champion_version(settings)
    if version is None:
        return None

    model = mlflow.sklearn.load_model(
        f"models:/{settings.registered_model_name}@{settings.model_alias}"
    )
    return ModelHandle(
        model=model,
        name=settings.registered_model_name,
        version=str(version.version),
        alias=settings.model_alias,
        run_id=getattr(version, "run_id", None),
    )


def register_and_promote(
    model: Any,
    input_example: Any,
    settings: Settings | None = None,
) -> ModelHandle:
    """Log the model to the active run, register it, and move the alias onto it.

    Must be called inside an active MLflow run.
    """
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    settings = settings or get_settings()

    # The signature declares what the service actually returns -- the probability
    # that team A wins -- rather than the 0/1 label sklearn's `predict` would give.
    signature = infer_signature(input_example, model.predict_proba(input_example)[:, 1])

    info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        input_example=input_example,
        signature=signature,
        # MLflow's default skops format refuses to serialise the boosted-tree
        # predictors this project uses. Models here are produced and consumed by
        # this repo's own pipelines, so cloudpickle is the appropriate format.
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        registered_model_name=settings.registered_model_name,
    )

    client = get_client(settings)
    version = _resolve_version(client, info, settings)
    client.set_registered_model_alias(settings.registered_model_name, settings.model_alias, version)
    logger.info(
        "promoted %s version %s to alias %r",
        settings.registered_model_name,
        version,
        settings.model_alias,
    )
    return ModelHandle(
        model=model,
        name=settings.registered_model_name,
        version=str(version),
        alias=settings.model_alias,
        run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None,
    )


def _resolve_version(client: Any, info: Any, settings: Settings) -> str:
    """Find the version number just created by ``log_model``.

    MLflow returns it on the ``ModelInfo`` in recent versions; older ones do
    not, so fall back to the newest version of the registered model.
    """
    version = getattr(info, "registered_model_version", None)
    if version is not None:
        return str(version)
    versions = client.search_model_versions(f"name='{settings.registered_model_name}'")
    if not versions:
        raise RuntimeError(f"no registered versions found for {settings.registered_model_name!r}")
    return str(max(int(v.version) for v in versions))


def should_promote(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    metric: str,
    higher_is_better: bool,
    min_improvement: float,
) -> tuple[bool, str]:
    """Decide whether the candidate replaces the champion.

    Returns the decision and a human-readable reason, which is logged to the
    MLflow run so the promotion history is auditable.
    """
    if champion is None:
        return True, "no champion registered yet"

    new, old = candidate[metric], champion[metric]
    improvement = (new - old) if higher_is_better else (old - new)
    if improvement >= min_improvement:
        return True, f"{metric} improved by {improvement:.4f} ({old:.4f} -> {new:.4f})"
    return False, f"{metric} did not improve enough: {improvement:.4f} < {min_improvement:.4f}"
