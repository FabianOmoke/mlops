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


def _build_pipeline(n_estimators: int = 200, excluded_features: list[str] | None = None) -> tuple[Pipeline, list[str], list[str]]:
    """
    Build a preprocessing + classification pipeline.
    
    Parameters
    ----------
    n_estimators : int
        Number of estimators for RandomForestClassifier (flow parameter for versioning).
    excluded_features : list[str] | None
        Feature names to exclude from training (e.g., ["lead_time"]).
    
    Returns
    -------
    pipeline, numeric_features, categorical_features
        The fitted Pipeline and the lists of features actually used.
    """
    if excluded_features is None:
        excluded_features = []
    
    used_numeric = [f for f in NUMERIC_FEATURES if f not in excluded_features]
    used_categorical = [f for f in CATEGORICAL_FEATURES if f not in excluded_features]
    
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, used_numeric),
        ("cat", categorical_transformer, used_categorical),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])
    return pipeline, used_numeric, used_categorical


def run_training(
    logger: Any = None,
    split_config: dict | None = None,
    excluded_features: list[str] | None = None,
    n_estimators: int = 200,
    robustness_threshold: float = 0.80,
) -> str:
    """
    Train a hotel cancellation classifier with configurable split and features.
    
    Parameters
    ----------
    logger : logging.Logger or Prefect logger
        Logger for output.
    split_config : dict | None
        Data split configuration: {"train": 0.7, "robustness": 0.1, "holdout": 0.2}
        If None, defaults to 80/20 (legacy Task 2 behavior).
    excluded_features : list[str] | None
        Features to exclude (e.g., ["lead_time"] for V2).
    n_estimators : int
        RandomForest estimator count (default 200 for V1 baseline).
    robustness_threshold : float
        Minimum ROC-AUC required at validation time (default 0.80).
    
    Returns
    -------
    str
        Model ID in registry.
    """
    log = logger or logging.getLogger(__name__)
    
    if split_config is None:
        split_config = {"train": 0.8, "robustness": 0.0, "holdout": 0.2}
    
    if excluded_features is None:
        excluded_features = []

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

    # Apply 3-way split
    train_ratio = split_config.get("train", 0.8)
    robustness_ratio = split_config.get("robustness", 0.0)
    holdout_ratio = split_config.get("holdout", 0.2)
    
    # Normalize in case they don't sum to 1
    total = train_ratio + robustness_ratio + holdout_ratio
    train_ratio /= total
    robustness_ratio /= total
    holdout_ratio /= total
    
    # First split: train + validation vs holdout
    X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
        X, y, test_size=holdout_ratio, random_state=42, stratify=y
    )
    
    # If robustness_ratio > 0, carve robustness_eval set
    if robustness_ratio > 0:
        robustness_frac = robustness_ratio / (train_ratio + robustness_ratio)
        X_train, X_robustness, y_train, y_robustness = train_test_split(
            X_train_val, y_train_val, test_size=robustness_frac, random_state=42, stratify=y_train_val
        )
    else:
        X_train, y_train = X_train_val, y_train_val
        X_robustness, y_robustness = None, None
    
    log.info(
        f"3-way split: train={len(X_train):,} | "
        f"robustness={len(X_robustness) if X_robustness is not None else 0:,} | "
        f"holdout={len(X_holdout):,}"
    )

    pipeline, used_numeric, used_categorical = _build_pipeline(
        n_estimators=n_estimators,
        excluded_features=excluded_features,
    )
    pipeline.fit(X_train, y_train)

    # Extract feature importances and map back to original numeric features
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        # ColumnTransformer provides the transformed feature names
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = []

    try:
        clf = pipeline.named_steps["classifier"]
        importances = clf.feature_importances_
    except Exception:
        importances = None

    numeric_importances: dict[str, float] = {}
    if importances is not None and feature_names:
        # Aggregate importances for each original numeric feature. Numeric
        # features are not expanded by OneHotEncoder, so names should match.
        for f in used_numeric:
            vals = [imp for name, imp in zip(feature_names, importances) if name.endswith(f) or name == f or f in name]
            numeric_importances[f] = float(sum(vals)) if vals else 0.0

    # Evaluate on robustness set if it exists, otherwise on holdout
    eval_X = X_robustness if X_robustness is not None else X_holdout
    eval_y = y_robustness if y_robustness is not None else y_holdout
    
    y_prob = pipeline.predict_proba(eval_X)[:, 1]
    auc = roc_auc_score(eval_y, y_prob)
    log.info(f"Training complete — evaluation ROC-AUC: {auc:.4f}")

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
            "numeric_features": used_numeric,
            "categorical_features": used_categorical,
        },
        output_schema={"target": TARGET, "type": "binary", "values": [0, 1]},
        dependencies=["scikit-learn", "joblib", "pandas", "pyarrow"],
        training_rows=len(X_train),
        test_rows=len(eval_X),
        split_config=split_config,
        excluded_features=excluded_features,
        n_estimators=n_estimators,
    )
    # Persist split row_id mapping for reproducibility / monitoring / A/B
    try:
        splits = {
            "train": list(df.loc[X_train.index, "row_id"].astype(str)),
            "robustness": list(df.loc[X_robustness.index, "row_id"].astype(str)) if X_robustness is not None else [],
            "holdout": list(df.loc[X_holdout.index, "row_id"].astype(str)),
        }
        splits_path = models_dir / f"splits_{model_id}.json"
        with open(splits_path, "w") as fh:
            import json

            json.dump(splits, fh, indent=2)
        log.info(f"Wrote split mapping → {splits_path}")
    except Exception:
        log.warning("Could not write split mapping (non-fatal)")

    # Register numeric feature importances as metadata (helpful for selecting drift features)
    if numeric_importances:
        registry.register(
            model_id=f"{model_id}-meta",
            artifact_path=final_path,
            metrics={},
            input_schema={},
            output_schema={},
            dependencies=[],
            numeric_feature_importances=numeric_importances,
        )
    log.info(f"Model registered with id={model_id}")
    return model_id
