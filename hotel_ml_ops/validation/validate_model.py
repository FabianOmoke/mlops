"""
validation/validate_model.py — Step 3: robustness validation.

Two checks are performed after loading the registered model:

Check 1 — Held-out ROC-AUC >= 0.80
-------------------------------------
Expectation: the model must achieve a ROC-AUC of at least 0.80 on a
freshly drawn held-out split (same random_state as training for
reproducibility, but evaluated here independently).

Reasoning: A dummy classifier predicting the majority class achieves
~0.50 AUC. Published baselines for this exact dataset (Antonio et al.,
2019; Kaggle leaderboard entries using tree ensembles) consistently
report AUC in the 0.87-0.93 range. We set 0.80 as the minimum acceptable
threshold — tight enough to catch a severely under-fit or data-leaked
model, but with enough headroom to allow for intentionally simplified
feature sets. Falling below 0.80 on ~24k test rows is a strong signal
of either a data issue, a training bug, or a leakage problem and must
block deployment.

Check 2 — Prediction stability under Gaussian noise (flip rate <= 5 %)
-----------------------------------------------------------------------
Expectation: adding small Gaussian noise (mean=0, sigma=0.01) to all
numeric features must not change more than 5 % of the binary predictions.

Reasoning: A production model receives data from a PMS/CRS that may
carry minor floating-point rounding errors or imputation artifacts. A
model that flips its prediction for > 5 % of inputs under sigma=0.01
perturbation is hyper-sensitive to irrelevant input variation — a sign
of over-fitting or an excessively sharp decision boundary. The 5 %
threshold was chosen to be stricter than random chance while allowing
the model a small tolerance for boundary cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "data" / "hotel_bookings.parquet"

NUMERIC_FEATURES = [
    "lead_time", "adr", "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "total_of_special_requests",
]
CATEGORICAL_FEATURES = [
    "hotel", "meal", "market_segment", "distribution_channel",
    "deposit_type", "customer_type", "reserved_room_type",
    "arrival_date_month",
]
TARGET = "is_canceled"
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

MIN_ROC_AUC = 0.80
MAX_FLIP_RATE = 0.05
NOISE_SIGMA = 0.01


def run_validation(model_id: str, logger: Any = None) -> None:
    log = logger or logging.getLogger(__name__)

    from versioning.model_registry import ModelRegistry
    registry = ModelRegistry(PROJECT_ROOT / "models" / "registry.json")
    pipeline = registry.load(model_id)
    log.info(f"Loaded model {model_id} for robustness validation.")

    df = pd.read_parquet(PARQUET_PATH)
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Check 1: ROC-AUC
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"Robustness check 1 — ROC-AUC: {auc:.4f} (threshold >= {MIN_ROC_AUC})")
    assert auc >= MIN_ROC_AUC, (
        f"ROBUSTNESS FAIL: ROC-AUC {auc:.4f} is below the minimum threshold of {MIN_ROC_AUC}."
    )
    log.info("  ✓ ROC-AUC check passed.")

    # Check 2: Prediction stability under noise
    X_test_numeric = X_test[NUMERIC_FEATURES].fillna(0).copy()
    rng = np.random.default_rng(seed=0)
    noise = rng.normal(loc=0.0, scale=NOISE_SIGMA, size=X_test_numeric.shape)

    X_test_noisy = X_test.copy()
    X_test_noisy[NUMERIC_FEATURES] = X_test_numeric.values + noise

    base_preds = pipeline.predict(X_test)
    noisy_preds = pipeline.predict(X_test_noisy)
    flip_rate = (base_preds != noisy_preds).mean()

    log.info(f"Robustness check 2 — Flip rate: {flip_rate:.4f} (threshold <= {MAX_FLIP_RATE})")
    assert flip_rate <= MAX_FLIP_RATE, (
        f"ROBUSTNESS FAIL: {flip_rate:.2%} of predictions flipped under noise, "
        f"exceeding the {MAX_FLIP_RATE:.0%} stability threshold."
    )
    log.info("  ✓ Prediction stability check passed.")
    log.info(f"Model {model_id} passed all robustness checks.")
