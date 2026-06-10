from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * (np.log(p) - np.log(q))))


def estimate_kl_threshold(reference_df: pd.DataFrame, features: list[str], n_bootstrap: int = 500, n_bins: int = 10, percentile: float = 95.0, logger: logging.Logger | None = None) -> Dict[str, float]:
    log = logger or logging.getLogger(__name__)
    thresholds: Dict[str, float] = {}
    rng = np.random.default_rng(seed=42)

    for feature in features:
        arr = reference_df[feature].dropna().to_numpy()
        if arr.size < 10:
            thresholds[feature] = 0.0
            continue
        kl_vals = []
        for _ in range(n_bootstrap):
            idx = rng.permutation(len(arr))
            a = arr[idx[: len(idx) // 2]]
            b = arr[idx[len(idx) // 2 :]]
            combined = np.concatenate([a, b])
            try:
                bin_edges = np.unique(np.quantile(combined, np.linspace(0, 1, n_bins + 1)))
                if len(bin_edges) < 2:
                    continue
                ha, _ = np.histogram(a, bins=bin_edges)
                hb, _ = np.histogram(b, bins=bin_edges)
                pa = (ha + 1e-10) / (ha.sum() + n_bins * 1e-10)
                pb = (hb + 1e-10) / (hb.sum() + n_bins * 1e-10)
                kl_vals.append(_kl(pa, pb))
            except Exception:
                continue
        thresholds[feature] = float(np.percentile(kl_vals, percentile)) if kl_vals else 0.0

    return thresholds


def compute_kl_divergences(reference_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], n_bins: int = 10) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for feature in features:
        ref = reference_df[feature].dropna().to_numpy()
        test = test_df[feature].dropna().to_numpy()
        if ref.size == 0 or test.size == 0:
            out[feature] = 0.0
            continue
        try:
            bin_edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
            if len(bin_edges) < 2:
                out[feature] = 0.0
                continue
            hr, _ = np.histogram(ref, bins=bin_edges)
            ht, _ = np.histogram(test, bins=bin_edges)
            pr = (hr + 1e-10) / (hr.sum() + n_bins * 1e-10)
            pt = (ht + 1e-10) / (ht.sum() + n_bins * 1e-10)
            out[feature] = float(_kl(pr, pt))
        except Exception:
            out[feature] = 0.0
    return out


def check_drift(reference_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], thresholds: dict[str, float], logger: logging.Logger | None = None) -> tuple[bool, dict]:
    log = logger or logging.getLogger(__name__)
    kl_vals = compute_kl_divergences(reference_df, test_df, features)
    results: dict = {}
    all_pass = True
    for f in features:
        kl = kl_vals.get(f, 0.0)
        thr = thresholds.get(f, float("inf"))
        passed = kl <= thr
        results[f] = {"kl": kl, "threshold": thr, "passed": passed}
        if not passed:
            all_pass = False
        status = "PASS" if passed else "DRIFT"
        log.info(f"{status} {f}: KL={kl:.6f} thr={thr:.6f}")
    return all_pass, results
