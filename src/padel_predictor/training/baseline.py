"""The provider's Elo as a model, so it can be registered and served like one.

A fresh registry starts with this model as its champion: until a trained model
beats the data provider's own Elo, the provider's Elo is what the system serves,
and a trained model has to win the same comparison every later candidate does
to replace it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: The name the baseline is logged under in the tracker.
PROVIDER_ELO_RUN_NAME = "provider-elo-baseline"


class ProviderEloModel:
    """Predicts ``provider_elo_prob``, the win probability implied by the provider's Elo.

    It reads the baseline column instead of :data:`FEATURE_COLUMNS`, which
    :func:`padel_predictor.common.features.model_inputs` picks up from
    ``input_columns``. A match the provider has no rating for gets 0.5, exactly
    as the baseline is scored in training.
    """

    input_columns: tuple[str, ...] = ("provider_elo_prob",)
    #: Fitted on no match, so every test match is unseen by it.
    trained_on_matches = False

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = pd.to_numeric(x["provider_elo_prob"], errors="coerce").to_numpy(dtype=float)
        p = np.where(np.isfinite(p), p, 0.5)
        return np.column_stack([1.0 - p, p])
