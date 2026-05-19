"""
training/train.py — Model training step for the hotel cancellation classifier.

Input schema
------------
Features (selected subset of hotel_bookings.parquet):
  Numeric  : lead_time, adr, stays_in_weekend_nights, stays_in_week_nights,
              adults, children, previous_cancellations,
              previous_bookings_not_canceled, booking_changes,
              days_in_waiting_list, total_of_special_requests
  Categorical (one-hot encoded): hotel, meal, market_segment,
              distribution_channel, deposit_type, customer_type,
              reserved_room_type, arrival_date_month

Output schema
-------------
  Binary label — is_canceled: 0 (kept) | 1 (canceled)
  Probability score — P(is_canceled=1) in [0, 1]

Model
-----
  sklearn RandomForestClassifier with 200 estimators.
  Chosen for interpretability (feature importances), robustness to
  missing/skewed numeric features, and strong out-of-the-box baseline
  performance on tabular data without hyperparameter tuning.

Serialization
-------------
  joblib is used (not pickle) because:
    - It handles numpy memory-mapped arrays efficiently.
    - Produces ~30-50 % smaller files than pickle for sklearn estimators.
    - Is the format explicitly recommended by scikit-learn for persistence.
  The full sklearn Pipeline (preprocessor + classifier) is serialized as a
  single artifact so the model is self-contained and no external preprocessing
  state is needed at inference time.

Error handling
--------------
  Small-dataset guard: if the training slice is < 1 000 rows (artificial
  injection via FORCE_SMALL_DATASET=1, or genuinely tiny data), a ValueError
  is raised immediately. This prevents silently producing an under-fit model
  that would pass accuracy checks by chance on a tiny test set.
  Rationale: for a binary classifier on a ~120k-row dataset a 1k training
  slice yields < 1 % of available signal — the resulting model would be
  statistically indistinguishable from random guessing on any meaningful
  held-out set.

  Interrupted-training guard: the serialization is written to a temp path
  first and atomically renamed on success, so a mid-write crash never leaves
  a corrupt artifact in the registry.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def _build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])


def run_training(logger: Any = None) -> str:
    log = logger or logging.getLogger(__name__)

    log.info(f"Loading data from {PARQUET_PATH} ...")
    df = pd.read_parquet(PARQUET_PATH)
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns])

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    force_small = os.environ.get("FORCE_SMALL_DATASET", "").strip() in {"1", "true", "yes"}
    if force_small:
        log.warning("FORCE_SMALL_DATASET=1 detected — truncating to 500 rows to trigger guard.")
        X = X.iloc[:500]
        y = y.iloc[:500]

    MIN_TRAINING_ROWS = 1_000
    if len(X) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Training aborted: only {len(X)} rows available, minimum is "
            f"{MIN_TRAINING_ROWS}. This indicates a data pipeline failure or "
            f"an accidental FORCE_SMALL_DATASET injection. Investigate upstream "
            f"before retraining."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Training on {len(X_train):,} rows, validating on {len(X_test):,} rows.")

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    log.info(f"Training complete — held-out ROC-AUC: {auc:.4f}")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    model_id = str(uuid.uuid4())[:8]
    final_path = models_dir / f"model_{model_id}.joblib"

    with tempfile.NamedTemporaryFile(dir=models_dir, suffix=".joblib.tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        joblib.dump(pipeline, tmp_path)
        tmp_path.rename(final_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    log.info(f"Model serialized → {final_path}")

    from versioning.model_registry import ModelRegistry
    registry = ModelRegistry(PROJECT_ROOT / "models" / "registry.json")
    registry.register(
        model_id=model_id,
        artifact_path=final_path,
        metrics={"roc_auc": round(auc, 4)},
        input_schema={
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        output_schema={"target": TARGET, "type": "binary", "values": [0, 1]},
        dependencies=["scikit-learn", "joblib", "pandas", "pyarrow"],
        training_rows=len(X_train),
        test_rows=len(X_test),
    )
    log.info(f"Model registered with id={model_id}")
    return model_id
