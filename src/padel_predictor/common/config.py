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

    # Feature store (local path or object-storage URI such as gs://bucket/prefix)
    feature_store_uri: str = Field(
        default="data/feature_store", validation_alias="PADEL_FEATURE_STORE_URI"
    )

    # Experiment tracking / model registry
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db", validation_alias="MLFLOW_TRACKING_URI"
    )
    mlflow_experiment_name: str = Field(
        default="padel-predictor", validation_alias="MLFLOW_EXPERIMENT_NAME"
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
            "feature_store_uri": self.feature_store_uri,
            "mlflow_tracking_uri": self.mlflow_tracking_uri,
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


class FeatureConfig(BaseModel):
    elo_initial: float = 1500.0
    elo_k_factor: float = 24.0
    form_window: int = 5
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


class TrainingConfig(BaseModel):
    test_fraction: float = 0.2
    min_training_rows: int = 200
    random_state: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    promotion: PromotionConfig = Field(default_factory=PromotionConfig)


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
