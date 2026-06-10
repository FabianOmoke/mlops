from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from prefect import flow, get_run_logger, task

from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@task(name="ab-compare")
def run_ab_test(flow_version_a: str, flow_version_b: str, experiment_id: str) -> dict:
    logger = get_run_logger()
    from versioning.flow_config import FlowConfigRegistry
    from versioning.model_registry import ModelRegistry
    from ab_testing.split_strategy import assign_variant

    fc = FlowConfigRegistry(PROJECT_ROOT / "models" / "flow_configs.json")
    mr = ModelRegistry(PROJECT_ROOT / "models" / "registry.json")

    a = fc.get(flow_version_a)
    b = fc.get(flow_version_b)
    if not a or not a.get("model_id"):
        raise KeyError(f"flow_version {flow_version_a} missing or unregistered")
    if not b or not b.get("model_id"):
        raise KeyError(f"flow_version {flow_version_b} missing or unregistered")

    model_a_id = a["model_id"]
    model_b_id = b["model_id"]

    logger.info(f"A: {flow_version_a} -> model {model_a_id}; B: {flow_version_b} -> model {model_b_id}")

    # load models
    model_a = mr.load(model_a_id)
    model_b = mr.load(model_b_id)

    # use holdout split from model_a (assume flows share the same holdout carving)
    splits_path = PROJECT_ROOT / "models" / f"splits_{model_a_id}.json"
    if not splits_path.is_file():
        raise FileNotFoundError(f"Holdout split not found for model {model_a_id}")
    with open(splits_path) as fh:
        splits = json.load(fh)

    holdout_ids = splits.get("holdout", [])
    if not holdout_ids:
        raise RuntimeError("No holdout rows available for A/B test")

    import pandas as pd

    df = pd.read_parquet(PROJECT_ROOT / "data" / "hotel_bookings.parquet")
    holdout_df = df[df["row_id"].astype(str).isin(holdout_ids)].copy()
    y_true = holdout_df["is_canceled"].values

    # assign variants deterministically
    variants = [assign_variant(r, experiment_id) for r in holdout_df["row_id"].astype(str)]
    holdout_df["variant"] = variants

    preds_a = model_a.predict_proba(holdout_df.drop(columns=["is_canceled"]))[:, 1]
    preds_b = model_b.predict_proba(holdout_df.drop(columns=["is_canceled"]))[:, 1]

    # compute metrics per variant
    import numpy as np

    results = {}
    for v in [0, 1]:
        mask = np.array(holdout_df["variant"]) == v
        if mask.sum() == 0:
            results[f"variant_{v}"] = {"roc_auc": None, "n": 0}
            continue
        if v == 0:
            roc = float(roc_auc_score(y_true[mask], preds_a[mask]))
        else:
            roc = float(roc_auc_score(y_true[mask], preds_b[mask]))
        results[f"variant_{v}"] = {"roc_auc": roc, "n": int(mask.sum())}

    logger.info(f"A/B results: {results}")
    return results


@flow(name="ab-test-flow")
def ab_test_flow(flow_version_a: str, flow_version_b: str, experiment_id: str) -> dict:
    return run_ab_test(flow_version_a, flow_version_b, experiment_id)


if __name__ == "__main__":
    # example runner (replace with real ids)
    import sys

    if len(sys.argv) < 4:
        print("Usage: python ab_test_flow.py <flow_version_a> <flow_version_b> <experiment_id>")
    else:
        ab_test_flow(sys.argv[1], sys.argv[2], sys.argv[3])
