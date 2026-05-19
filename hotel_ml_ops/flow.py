"""
Task 2 — Pre-deployment pipeline (Prefect local runner).

Steps
-----
1. data_quality_step  : runs the Great Expectations checks from Task 1.
2. training_step      : trains a RandomForest classifier, serializes the
                        artifact with joblib, and registers it in the local
                        model registry.
3. robustness_step    : loads the just-registered model and validates that
                        its held-out ROC-AUC meets a minimum benchmark and
                        that predictions are stable under light input noise.

Run
---
    cd hotel_ml_ops
    python flow.py

Error injection
---------------
    FORCE_SMALL_DATASET=1 python flow.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from prefect import flow, task, get_run_logger

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


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
def training_step() -> str:
    logger = get_run_logger()
    from training.train import run_training
    model_id = run_training(logger=logger)
    return model_id


@task(name="robustness-validation")
def robustness_step(model_id: str) -> None:
    logger = get_run_logger()
    from validation.validate_model import run_validation
    run_validation(model_id=model_id, logger=logger)


@flow(name="hotel-ml-pre-deployment", log_prints=True)
def hotel_ml_pipeline() -> None:
    data_quality_step()
    model_id = training_step()
    robustness_step(model_id)


if __name__ == "__main__":
    hotel_ml_pipeline()
