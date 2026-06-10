"""
Task 2 & Task 3 — Pre-deployment and versioning pipeline (Prefect local runner).

Steps
-----
1. data_quality_step  : runs the Great Expectations checks from Task 1.
2. training_step      : trains a RandomForest classifier, serializes the
                        artifact with joblib, and registers it in the local
                        model registry.
3. robustness_step    : loads the just-registered model and validates that
                        its held-out ROC-AUC meets a minimum benchmark and
                        that predictions are stable under light input noise.

Flow Configuration (for Task 3 versioning)
-------------------------------------------
The flow now accepts parameters for versioning different model configurations:

  split_config : dict
    Data split ratios. Default: {train: 0.7, robustness: 0.1, holdout: 0.2}
    - train (70%): used for model training
    - robustness (10%): used for pre-deployment robustness testing
    - holdout (20%): held completely unseen; used for Task 3 monitoring + A/B

  excluded_features : list[str]
    Features to exclude from training (e.g., ["lead_time"] for V2).
    Default: [] (use all features).

  n_estimators : int
    Number of estimators in RandomForestClassifier. Default: 200.

  robustness_threshold : float
    Minimum ROC-AUC to pass validation. Default: 0.80.

Flow Versions
-------------
V1 (Baseline): split_config={train: 0.7, robustness: 0.1, holdout: 0.2},
                excluded_features=[], n_estimators=200
V2 (Modified):  split_config={train: 0.7, robustness: 0.1, holdout: 0.2},
                excluded_features=["lead_time"], n_estimators=100

Run
---
    cd hotel_ml_ops
    python flow.py  # V1 baseline with new 3-way split
    
    # Or with custom config (e.g., V2):
    FLOW_SPLIT_CONFIG='{"train":0.7,"robustness":0.1,"holdout":0.2}' \\
    FLOW_EXCLUDED_FEATURES='lead_time' \\
    FLOW_N_ESTIMATORS='100' \\
    python flow.py

Error injection
---------------
    FORCE_SMALL_DATASET=1 python flow.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_flow_config() -> dict:
    """Parse flow configuration from environment or defaults."""
    split_config_str = os.environ.get("FLOW_SPLIT_CONFIG")
    split_config = json.loads(split_config_str) if split_config_str else {
        "train": 0.7,
        "robustness": 0.1,
        "holdout": 0.2,
    }
    
    excluded_str = os.environ.get("FLOW_EXCLUDED_FEATURES", "").strip()
    excluded_features = [f.strip() for f in excluded_str.split(",") if f.strip()]
    
    n_estimators_str = os.environ.get("FLOW_N_ESTIMATORS", "200").strip()
    n_estimators = int(n_estimators_str)
    
    robustness_threshold_str = os.environ.get("FLOW_ROBUSTNESS_THRESHOLD", "0.80").strip()
    robustness_threshold = float(robustness_threshold_str)
    
    return {
        "split_config": split_config,
        "excluded_features": excluded_features,
        "n_estimators": n_estimators,
        "robustness_threshold": robustness_threshold,
    }


def _flow_version_id() -> str:
    vid = os.environ.get("FLOW_VERSION_ID")
    if vid:
        return vid
    # fallback: short uuids to identify this run
    import uuid

    return str(uuid.uuid4())[:8]


@task(name="data-quality-tests", retries=1, retry_delay_seconds=5)
def data_quality_step() -> None:
    logger = get_run_logger()
    logger.info("Running pre-training data quality tests ...")

    from tests.test_data_quality import (
        test_row_count_non_trivial,
        test_mixed_attribute_columns_present,
        test_hotel_data,
    )

    test_row_count_non_trivial()
    logger.info("  ✓ row count non-trivial")

    test_mixed_attribute_columns_present()
    logger.info("  ✓ mixed attribute columns present")

    test_hotel_data()
    logger.info("  ✓ GE distribution / null checks passed")

    logger.info("All data quality tests passed — proceeding to training.")


@task(name="model-training")
def training_step(flow_config: dict) -> str:
    logger = get_run_logger()
    from training.train import run_training
    model_id = run_training(
        logger=logger,
        split_config=flow_config["split_config"],
        excluded_features=flow_config["excluded_features"],
        n_estimators=flow_config["n_estimators"],
        robustness_threshold=flow_config["robustness_threshold"],
    )
    return model_id


@task(name="robustness-validation")
def robustness_step(model_id: str, flow_config: dict) -> None:
    logger = get_run_logger()
    from validation.validate_model import run_validation
    run_validation(
        model_id=model_id,
        logger=logger,
        min_roc_auc=flow_config["robustness_threshold"],
    )


@flow(name="hotel-ml-pre-deployment", log_prints=True)
def hotel_ml_pipeline() -> None:
    flow_config = _parse_flow_config()
    logger = get_run_logger()
    
    logger.info(f"Flow configuration:")
    logger.info(f"  split_config: {flow_config['split_config']}")
    logger.info(f"  excluded_features: {flow_config['excluded_features']}")
    logger.info(f"  n_estimators: {flow_config['n_estimators']}")
    logger.info(f"  robustness_threshold: {flow_config['robustness_threshold']}")
    
    data_quality_step()
    model_id = training_step(flow_config)
    robustness_step(model_id, flow_config)

    # Register flow configuration → model id mapping for versioning / A/B
    from versioning.flow_config import FlowConfigRegistry

    flow_vid = _flow_version_id()
    fc = FlowConfigRegistry(PROJECT_ROOT / "models" / "flow_configs.json")
    fc.register(flow_vid, flow_config, model_id=model_id)
    logger.info(f"Registered flow version {flow_vid} -> model {model_id}")


if __name__ == "__main__":
    hotel_ml_pipeline()
