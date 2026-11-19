"""Experiment tracking and model registry.

Every tracking call in the project goes through this module. The backend is
chosen by ``PADEL_TRACKING``:

``wandb``
    Weights & Biases, the production choice. A training run is a W&B run
    (config, metrics, summary); every candidate is logged as a ``model``
    artifact named after ``PADEL_REGISTERED_MODEL_NAME``, and the promoted one
    carries the ``champion`` alias. W&B keeps an alias unique within an
    artifact, so promoting a new version moves it off the old one.
``local``
    The same contract on disk under ``PADEL_MODEL_REGISTRY_DIR``, for the tests
    and an offline clone without a W&B account.

Models are addressed by *alias* (``champion``), so the inference pipeline asks
for "the best model" and never hard-codes a version.

Every model also records ``train_end``, the last match it was trained on. The
test window moves whenever history is backfilled, so a champion trained months
ago can have seen matches that are in today's test set; the training pipeline
uses ``train_end`` to score the champion only on matches it has never seen.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import joblib
import pandas as pd

from padel_predictor.common.config import Settings, get_settings
from padel_predictor.common.logging import get_logger

logger = get_logger(__name__)

MODEL_FILENAME = "model.joblib"


@dataclass(frozen=True)
class ModelHandle:
    """A loaded model plus the provenance the UI and API report."""

    model: Any
    name: str
    version: str
    alias: str
    run_id: str | None = None
    #: The last match the model was trained on; ``None`` when it was not recorded.
    train_end: pd.Timestamp | None = None

    @property
    def uri(self) -> str:
        return f"{self.name}:{self.version}"


def parse_train_end(value: Any) -> pd.Timestamp | None:
    """Read a recorded training end back as a UTC timestamp, or ``None``."""
    if isinstance(value, dict):  # W&B configs can wrap values as {"value": ...}
        value = value.get("value")
    if value is None or value == "":
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        logger.warning("could not read train_end %r", value)
        return None
    if pd.isna(stamp):
        return None
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


class TrainingRun(Protocol):
    """One tracked training run, whichever backend records it."""

    run_id: str

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def set_tags(self, tags: dict[str, str]) -> None: ...

    def log_model(self, model: Any, promote: bool, metadata: dict[str, Any]) -> ModelHandle: ...


def _backend(settings: Settings) -> str:
    backend = settings.tracking.strip().lower()
    if backend not in {"wandb", "local"}:
        raise ValueError(f"PADEL_TRACKING must be 'wandb' or 'local', got {settings.tracking!r}")
    return backend


@contextmanager
def start_run(
    settings: Settings | None = None, name: str = "training", job_type: str | None = None
):
    """Open a tracked run; it is closed (and flushed) when the block exits.

    ``job_type`` groups runs in W&B (``training``, ``ablation``); it defaults
    to the run's name.
    """
    settings = settings or get_settings()
    run: _WandbRun | _LocalRun = (
        _WandbRun(settings, name, job_type or name)
        if _backend(settings) == "wandb"
        else _LocalRun(settings, name)
    )
    try:
        yield run
    finally:
        run.finish()


def load_champion(settings: Settings | None = None) -> ModelHandle | None:
    """Load the model carrying the champion alias, or ``None`` if there is none."""
    settings = settings or get_settings()
    if _backend(settings) == "wandb":
        return _wandb_load(settings)
    return _local_load(settings)


def register_first_champion(
    settings: Settings, model: Any, run_name: str, params: dict[str, Any]
) -> ModelHandle:
    """Log an untrained model (the provider's Elo) as the champion of an empty registry."""
    with start_run(settings, name=run_name) as run:
        run.log_params(params)
        run.set_tags({"promoted": "True", "promotion_reason": "first champion"})
        handle = run.log_model(model, promote=True, metadata={"promotion_reason": "first champion"})
    logger.info("registered %s as the first champion (%s)", run_name, handle.uri)
    return handle


def should_promote(
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    metric: str,
    higher_is_better: bool,
    min_improvement: float,
) -> tuple[bool, str]:
    """Decide whether the candidate replaces the champion.

    Returns the decision and a human-readable reason, which is logged with the
    run so the promotion history is auditable.
    """
    if champion is None:
        return True, "no champion registered yet"

    new, old = candidate[metric], champion[metric]
    improvement = (new - old) if higher_is_better else (old - new)
    if improvement >= min_improvement:
        return True, f"{metric} improved by {improvement:.4f} ({old:.4f} -> {new:.4f})"
    return False, f"{metric} did not improve enough: {improvement:.4f} < {min_improvement:.4f}"


# --------------------------------------------------------------------------- #
# Weights & Biases
# --------------------------------------------------------------------------- #
def _wandb_path(settings: Settings, api: Any, alias: str) -> str:
    entity = settings.wandb_entity or api.default_entity
    return f"{entity}/{settings.wandb_project}/{settings.registered_model_name}:{alias}"


class _WandbRun:
    def __init__(self, settings: Settings, name: str, job_type: str) -> None:
        import wandb

        self.settings = settings
        self.run = wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity or None,
            job_type=job_type,
            name=name,
        )
        self.run_id = self.run.id

    def log_params(self, params: dict[str, Any]) -> None:
        self.run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.run.log(metrics)
        self.run.summary.update(metrics)

    def set_tags(self, tags: dict[str, str]) -> None:
        self.run.summary.update({f"tag/{k}": v for k, v in tags.items()})
        if tags.get("promoted") == "True":
            self.run.tags = (*self.run.tags, "promoted")
        if tags.get("experiment"):
            self.run.tags = (*self.run.tags, tags["experiment"])
        if tags.get("data_plan"):
            self.run.tags = (*self.run.tags, f"plan-{tags['data_plan']}")

    def log_model(self, model: Any, promote: bool, metadata: dict[str, Any]) -> ModelHandle:
        import wandb

        name = self.settings.registered_model_name
        artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / MODEL_FILENAME
            joblib.dump(model, path)
            artifact.add_file(str(path))
            aliases = ["latest", self.settings.model_alias] if promote else ["latest"]
            logged = self.run.log_artifact(artifact, aliases=aliases)
            if getattr(self.run, "offline", False):
                # W&B assigns versions server-side; offline runs only sync later.
                version = "offline"
            else:
                logged.wait()
                version = str(logged.version)
        logger.info("logged %s:%s (aliases %s)", name, version, aliases)
        return ModelHandle(
            model=model,
            name=name,
            version=version,
            alias=self.settings.model_alias if promote else "latest",
            run_id=self.run_id,
        )

    def finish(self) -> None:
        self.run.finish()


def _wandb_load(settings: Settings) -> ModelHandle | None:
    import wandb

    api = wandb.Api()
    path = _wandb_path(settings, api, settings.model_alias)
    try:
        artifact = api.artifact(path, type="model")
    except Exception as exc:  # noqa: BLE001 -- a missing alias is the normal first-run case
        logger.info("no model at %s yet (%s)", path, type(exc).__name__)
        return None
    with tempfile.TemporaryDirectory() as tmp:
        model = joblib.load(Path(artifact.download(root=tmp)) / MODEL_FILENAME)
    producer = artifact.logged_by()
    # Models logged before train_end was part of the artifact metadata still
    # have it in the config of the run that trained them.
    train_end = (artifact.metadata or {}).get("train_end")
    if train_end is None and producer is not None:
        train_end = (producer.config or {}).get("train_end")
    return ModelHandle(
        model=model,
        name=settings.registered_model_name,
        version=str(artifact.version),
        alias=settings.model_alias,
        run_id=producer.id if producer is not None else None,
        train_end=parse_train_end(train_end),
    )


# --------------------------------------------------------------------------- #
# Local registry (offline, tests)
# --------------------------------------------------------------------------- #
#   <dir>/runs/<run_id>.json
#   <dir>/<model>/v<N>/model.joblib + metadata.json
#   <dir>/<model>/aliases.json        {"champion": "v3"}
@dataclass
class _LocalRun:
    settings: Settings
    name: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    record: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root = Path(self.settings.local_registry_dir)
        self.record = {
            "run_id": self.run_id,
            "name": self.name,
            "started_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "params": {},
            "metrics": {},
            "tags": {},
        }

    def log_params(self, params: dict[str, Any]) -> None:
        self.record["params"].update({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.record["metrics"].update({k: float(v) for k, v in metrics.items()})

    def set_tags(self, tags: dict[str, str]) -> None:
        self.record["tags"].update(tags)

    def log_model(self, model: Any, promote: bool, metadata: dict[str, Any]) -> ModelHandle:
        model_dir = self.root / self.settings.registered_model_name
        existing = [int(p.name[1:]) for p in model_dir.glob("v*") if p.name[1:].isdigit()]
        version = f"v{max(existing, default=0) + 1}"
        target = model_dir / version
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target / MODEL_FILENAME)
        (target / "metadata.json").write_text(
            json.dumps({**metadata, "run_id": self.run_id}, indent=2, default=str)
        )
        if promote:
            aliases_file = model_dir / "aliases.json"
            aliases = json.loads(aliases_file.read_text()) if aliases_file.is_file() else {}
            aliases[self.settings.model_alias] = version
            aliases_file.write_text(json.dumps(aliases, indent=2))
        self.record["model_version"] = version
        return ModelHandle(
            model=model,
            name=self.settings.registered_model_name,
            version=version,
            alias=self.settings.model_alias if promote else "latest",
            run_id=self.run_id,
        )

    def finish(self) -> None:
        runs = self.root / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{self.run_id}.json").write_text(json.dumps(self.record, indent=2, default=str))


def _local_load(settings: Settings) -> ModelHandle | None:
    model_dir = Path(settings.local_registry_dir) / settings.registered_model_name
    aliases_file = model_dir / "aliases.json"
    if not aliases_file.is_file():
        logger.info("no %r alias under %s yet", settings.model_alias, model_dir)
        return None
    version = json.loads(aliases_file.read_text()).get(settings.model_alias)
    if version is None:
        return None
    metadata = json.loads((model_dir / version / "metadata.json").read_text())
    train_end = metadata.get("train_end")
    run_file = Path(settings.local_registry_dir) / "runs" / f"{metadata.get('run_id')}.json"
    if train_end is None and run_file.is_file():
        train_end = json.loads(run_file.read_text()).get("params", {}).get("train_end")
    return ModelHandle(
        model=joblib.load(model_dir / version / MODEL_FILENAME),
        name=settings.registered_model_name,
        version=version,
        alias=settings.model_alias,
        run_id=metadata.get("run_id"),
        train_end=parse_train_end(train_end),
    )
