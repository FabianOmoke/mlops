from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from prefect import flow, task, get_run_logger

from monitoring.drift_utils import estimate_kl_threshold, check_drift

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@task(name="feature-importance-extraction")
def feature_importance_step() -> list[str]:
    logger = get_run_logger()
    from versioning.model_registry import ModelRegistry
    from versioning.flow_config import FlowConfigRegistry

    # Choose a baseline flow version (expectation: registered by flow runs)
    fc = FlowConfigRegistry(PROJECT_ROOT / "models" / "flow_configs.json")
    mr = ModelRegistry(PROJECT_ROOT / "models" / "registry.json")
    entries = fc.list()
    if not entries:
        raise RuntimeError("No flow versions registered; run training flow first")
    # pick the first registered flow as baseline
    baseline_vid = list(entries.keys())[0]
    baseline = fc.get(baseline_vid)
    model_id = baseline.get("model_id")

    pipeline = mr.load(model_id)
    logger.info(f"Loaded baseline model {model_id} for feature importance.")

    rf = pipeline.named_steps["classifier"]
    pre = pipeline.named_steps["preprocessor"]
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = []

    import pandas as pd

    fi = getattr(rf, "feature_importances_", None)
    if fi is None or len(feature_names) == 0:
        # fallback to numeric features list
        numeric_features = [
            "lead_time",
            "adr",
            "stays_in_weekend_nights",
            "stays_in_week_nights",
            "adults",
            "children",
            "previous_cancellations",
            "previous_bookings_not_canceled",
            "booking_changes",
            "days_in_waiting_list",
            "total_of_special_requests",
        ]
        logger.info("Falling back to default numeric feature list for monitoring")
        return numeric_features[:4]

    df = pd.DataFrame({"feature": feature_names, "importance": fi}).sort_values("importance", ascending=False)
    numeric_features = [
        "lead_time",
        "adr",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "booking_changes",
        "days_in_waiting_list",
        "total_of_special_requests",
    ]
    top_numeric = []
    for feat in df["feature"]:
        for orig in numeric_features:
            if orig in feat and orig not in top_numeric:
                top_numeric.append(orig)
                break
        if len(top_numeric) >= 4:
            break

    logger.info(f"Top numeric features: {top_numeric}")
    return top_numeric


@task(name="bootstrap-threshold-estimation")
def bootstrap_threshold_step(features: list[str]) -> dict[str, float]:
    logger = get_run_logger()
    df = __import__("pandas").read_parquet(PROJECT_ROOT / "data" / "hotel_bookings.parquet")

    # Recreate reproducible 70/10/20 split used by training flows
    from sklearn.model_selection import train_test_split

    X_train_val, X_holdout = train_test_split(df, test_size=0.2, random_state=42)
    robustness_frac = 0.1 / 0.8
    X_train, X_robustness = train_test_split(X_train_val, test_size=robustness_frac, random_state=42)

    logger.info(f"Estimating thresholds on {len(X_train):,} training rows")
    thresholds = estimate_kl_threshold(X_train, features, n_bootstrap=500, n_bins=10)
    return thresholds


@task(name="drift-check")
def drift_check_step(features: list[str], thresholds: dict[str, float]) -> dict:
    logger = get_run_logger()
    df = __import__("pandas").read_parquet(PROJECT_ROOT / "data" / "hotel_bookings.parquet")
    from sklearn.model_selection import train_test_split

    X_train_val, X_holdout = train_test_split(df, test_size=0.2, random_state=42)
    robustness_frac = 0.1 / 0.8
    X_train, X_robustness = train_test_split(X_train_val, test_size=robustness_frac, random_state=42)

    no_drift, results = check_drift(X_train, X_holdout, features, thresholds, logger=logger)
    return results


@flow(name="hotel-ml-monitoring", log_prints=True)
def monitoring_pipeline() -> None:
    logger = get_run_logger()
    logger.info("Starting monitoring flow")
    features = feature_importance_step()
    thresholds = bootstrap_threshold_step(features)
    results = drift_check_step(features, thresholds)
    logger.info("Monitoring flow complete")


if __name__ == "__main__":
    monitoring_pipeline()
