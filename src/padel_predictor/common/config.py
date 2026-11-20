"""Configuration.

Two sources, deliberately separated:

* **Secrets and environment-specific URIs** come from environment variables
  (``.env`` locally, GitHub Actions secrets in CI). See ``.env.example``.
* **Pipeline behaviour** (feature parameters, model hyper-parameters, promotion
  rules) comes from ``config/settings.yaml``, which is committed and contains
  no secrets.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]


# --------------------------------------------------------------------------- #
# Environment (secrets, URIs)
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Environment-provided settings. Never hard-code values here."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Data source
    source: str = Field(default="sample", validation_alias="PADEL_SOURCE")
    api_base_url: str = Field(default="", validation_alias="PADEL_API_BASE_URL")
    api_key: SecretStr = Field(default=SecretStr(""), validation_alias="PADEL_API_KEY")
    # "paid": full archive and ranking history. "free": last six months and the
    # current ranking only -- the fallback for anyone without a subscription.
    api_plan: str = Field(default="paid", validation_alias="PADEL_API_PLAN")

    # Feature store: hopsworks://<project> in production; a local path or a
    # gs:// prefix selects the Parquet store used offline and in the tests.
    feature_store_uri: str = Field(
        default="data/feature_store", validation_alias="PADEL_FEATURE_STORE_URI"
    )
    hopsworks_api_key: SecretStr = Field(
        default=SecretStr(""), validation_alias="HOPSWORKS_API_KEY"
    )
    # Empty means the hosted serverless app (the client library's default host).
    hopsworks_host: str = Field(default="", validation_alias="HOPSWORKS_HOST")

    # Experiment tracking / model registry: "wandb" in production, "local" offline.
    tracking: str = Field(default="local", validation_alias="PADEL_TRACKING")
    wandb_project: str = Field(default="padel-predictor", validation_alias="WANDB_PROJECT")
    wandb_entity: str = Field(default="", validation_alias="WANDB_ENTITY")
    wandb_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="WANDB_API_KEY")
    local_registry_dir: str = Field(
        default="data/model_registry", validation_alias="PADEL_MODEL_REGISTRY_DIR"
    )
    registered_model_name: str = Field(
        default="padel-match-winner", validation_alias="PADEL_REGISTERED_MODEL_NAME"
    )
    model_alias: str = Field(default="champion", validation_alias="PADEL_MODEL_ALIAS")

    # Serving
    api_host: str = Field(default="0.0.0.0", validation_alias="PADEL_API_HOST")
    api_port: int = Field(default=8000, validation_alias="PADEL_API_PORT")
    api_url: str = Field(default="http://localhost:8000", validation_alias="PADEL_API_URL")

    def masked(self) -> dict[str, str]:
        """A log-safe view of the settings: secrets are reduced to set/unset."""
        return {
            "source": self.source,
            "api_base_url": self.api_base_url or "<unset>",
            "api_key": "<set>" if self.api_key.get_secret_value() else "<unset>",
            "api_plan": self.api_plan,
            "feature_store_uri": self.feature_store_uri,
            "hopsworks_api_key": "<set>"
            if self.hopsworks_api_key.get_secret_value()
            else "<unset>",
            "tracking": self.tracking,
            "wandb_project": self.wandb_project,
            "wandb_api_key": "<set>" if self.wandb_api_key.get_secret_value() else "<unset>",
            "registered_model_name": self.registered_model_name,
            "model_alias": self.model_alias,
        }


# --------------------------------------------------------------------------- #
# settings.yaml (pipeline behaviour)
# --------------------------------------------------------------------------- #
class IngestionConfig(BaseModel):
    lookback_days: int = 7
    timeout_seconds: float = 30.0
    max_retries: int = 3
    # Ranking snapshots are re-read this many days before a window, so the
    # snapshot in force at the window's first match is always in the store.
    rankings_lookback_days: int = 14
    # How far back the free plan reaches.
    free_plan_history_days: int = 183
    # padelapi.org tournament levels read by the http source.
    levels: list[str] = Field(
        default_factory=lambda: ["major", "p1", "p2", "finals", "fip_platinum", "fip_gold"]
    )
    # Results per page (the provider's maximum is 50).
    page_size: int = 50
    # Requests are paced to stay under the plan's per-minute limit.
    requests_per_minute: dict[str, int] = Field(default_factory=lambda: {"free": 9, "paid": 55})
    # Pages of 50 read per ranking list, category and week (4 = top 200).
    rankings_max_pages: int = 4
    # Paid plan: read each player of the fetched matches who is missing from
    # those lists on their own (one request per player and window), at most
    # this many players per window.
    rankings_fill_missing_players: bool = True
    rankings_max_player_requests: int = 400
    # A longer Retry-After than this means the daily quota is gone: stop.
    max_retry_after_seconds: float = 120.0


class FeatureConfig(BaseModel):
    elo_initial: float = 1500.0
    elo_k_factor: float = 24.0
    form_window: int = 5
    # A ranking snapshot counts as published this many days after its label
    # date, in a backfill and in a live run alike. None joins on the
    # publication time as stored.
    rankings_publication_lag_days: float | None = 7.0
    tier_levels: dict[str, int] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    max_iter: int = 300
    learning_rate: float = 0.06
    max_depth: int | None = 4
    l2_regularization: float = 1.0
    early_stopping: bool = True


class PromotionConfig(BaseModel):
    metric: str = "log_loss"
    higher_is_better: bool = False
    min_improvement: float = 0.002
    # The champion is only scored on test matches it was not trained on; with
    # fewer than this many, there is no fair comparison and nothing is promoted.
    min_comparison_rows: int = 100
    # An empty registry gets the provider's Elo as its first champion, so a
    # trained model is only served once it beats the provider.
    provider_elo_first_champion: bool = True


class EvaluationConfig(BaseModel):
    bootstrap_samples: int = 2000
    confidence: float = 0.95


class AblationConfig(BaseModel):
    # Features dropped one at a time by `padel ablation`.
    drop: list[str] = Field(
        default_factory=lambda: ["partnership_diff", "form_diff", "rest_days_diff"]
    )


class TrainingWindowConfig(BaseModel):
    # `padel training-window` fits only on the last N months before the test set.
    months: list[int] = Field(default_factory=lambda: [30, 24, 18, 12, 6])


class TrainingConfig(BaseModel):
    test_fraction: float = 0.2
    min_training_rows: int = 200
    random_state: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    training_window: TrainingWindowConfig = Field(default_factory=TrainingWindowConfig)


class InferenceConfig(BaseModel):
    horizon_days: int = 7


class PipelineConfig(BaseModel):
    """Typed view of ``config/settings.yaml``."""

    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)


def _settings_file() -> Path | None:
    """Locate ``settings.yaml``: explicit override, then CWD, then repo root."""
    override = os.getenv("PADEL_SETTINGS_FILE")
    candidates = [Path(override)] if override else []
    candidates += [Path.cwd() / "config" / "settings.yaml", REPO_ROOT / "config" / "settings.yaml"]
    return next((c for c in candidates if c.is_file()), None)


def load_pipeline_config(path: Path | None = None) -> PipelineConfig:
    """Load and validate the pipeline configuration.

    Falls back to the in-code defaults when no ``settings.yaml`` is found, so the
    package stays importable when installed as a wheel without the repo.
    """
    resolved = path or _settings_file()
    if resolved is None:
        return PipelineConfig()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return PipelineConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached environment settings."""
    return Settings()


@lru_cache(maxsize=1)
def get_pipeline_config() -> PipelineConfig:
    """Cached pipeline configuration."""
    return load_pipeline_config()


def reset_caches() -> None:
    """Clear cached configuration (used by tests that patch the environment)."""
    get_settings.cache_clear()
    get_pipeline_config.cache_clear()
