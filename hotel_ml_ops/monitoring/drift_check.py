"""Monitoring utilities: bootstrap-derived KL thresholds and drift checks."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * (np.log(p) - np.log(q))))


def estimate_kl_threshold(
    reference_df: pd.DataFrame,
    features: list[str],
    n_bootstrap: int = 500,
    percentile: float = 95.0,
    n_bins: int = 10,
    logger: logging.Logger | None = None,
) -> Dict[str, float]:
    log = logger or logging.getLogger(__name__)
    thresholds: Dict[str, float] = {}
    rng = np.random.default_rng(seed=42)

    for feature in features:
        kl_vals = []
        arr = reference_df[feature].dropna().to_numpy()
        if arr.size < 10:
            log.warning(f"Not enough data for bootstrap on {feature}")
            thresholds[feature] = 0.0
            continue

        for _ in range(n_bootstrap):
            idx = rng.permutation(len(arr))
            a = arr[idx[: len(idx) // 2]]
            b = arr[idx[len(idx) // 2 :]]

            combined = np.concatenate([a, b])
            try:
                quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = np.unique(np.quantile(combined, quantiles))
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
"""Monitoring utilities for bootstrap KL-based drift detection."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Dict


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * (np.log(p) - np.log(q))))


def estimate_kl_threshold(
    reference_df: pd.DataFrame,
    features: list[str],
    n_bootstrap: int = 500,
    percentile: float = 95.0,
    n_bins: int = 10,
    logger: logging.Logger | None = None,
) -> Dict[str, float]:
    log = logger or logging.getLogger(__name__)
    thresholds: Dict[str, float] = {}
    rng = np.random.default_rng(seed=42)

    for feature in features:
        kl_vals = []
        arr = reference_df[feature].dropna().to_numpy()
        if arr.size < 10:
            log.warning(f"Not enough data for bootstrap on {feature}")
            thresholds[feature] = 0.0
            continue

        for _ in range(n_bootstrap):
            idx = rng.permutation(len(arr))
            a = arr[idx[: len(idx) // 2]]
            b = arr[idx[len(idx) // 2 :]]

            combined = np.concatenate([a, b])
            try:
                quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = np.unique(np.quantile(combined, quantiles))
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
"""monitoring/drift_check.py — Feature distribution drift detection.

Detects distribution drift in production data vs. training reference using
KL divergence on key numeric features. The threshold is **empirically derived**
from training data bootstrapping, not a fixed constant.

Bootstrap Threshold Derivation (Data-Sourced Expectation)
----------------------------------------------------------
We resample the training data ~1000 times into random 50/50 subsamples,
computing KL(subsample_A || subsample_B) for each feature. The 95th
percentile of these bootstrap KL values becomes our "natural noise floor"—
the drift amount we expect from random sampling variation alone.

**Why bootstrap?**
  - KL divergence is unbounded and scale-dependent (no universal threshold).
  - By deriving the threshold from the data itself, we anchor it to the
    actual feature distributions and binning scheme.
  - The 95th percentile is a defensible choice: it allows genuine natural
    variation while catching systematic shifts.

**Feature Selection:**
  - Restricted to top 3–4 **numeric** features by model feature importance.
  - Rationale: (1) Feature importances in sklearn pipelines are indexed by
    post-transform columns (post-OneHotEncoder), so categorical importances
    don't map 1:1 back to original features; (2) KL divergence is naturally
    suited for continuous distributions (what we bin and compute); (3) top
    features ensure we monitor the most impactful variables.

**Binning Strategy:**
  - Quantile-based (10 bins per feature) ensures equal-count bins and
    robustness to outliers. Avoids extreme bin widths that plague fixed-width
    binning on skewed data.

References
----------
- KL divergence: D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
  (asymmetric; measures how Q diverges from P's perspective)
- Quantile binning: pd.qcut() ensures each bin has ~equal count.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * (np.log(p) - np.log(q))))


def estimate_kl_threshold(
    reference_df: pd.DataFrame,
    features: list[str],
    n_bootstrap: int = 1000,
    percentile: float = 95.0,
    n_bins: int = 10,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    log = logger or logging.getLogger(__name__)
    thresholds: dict[str, float] = {}
    rng = np.random.default_rng(seed=42)

    for feature in features:
        kl_vals = []
        arr = reference_df[feature].dropna().to_numpy()
        if arr.size < 10:
            log.warning(f"Not enough data for bootstrap on {feature}")
            thresholds[feature] = 0.0
            continue

        for _ in range(n_bootstrap):
            # make two disjoint 50/50 subsamples
            idx = rng.permutation(len(arr))
            a = arr[idx[: len(idx) // 2]]
            b = arr[idx[len(idx) // 2 :]]

            combined = np.concatenate([a, b])
            try:
                quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = np.unique(np.quantile(combined, quantiles))
                if len(bin_edges) < 2:
                    continue
                ha, _ = np.histogram(a, bins=bin_edges)
                hb, _ = np.histogram(b, bins=bin_edges)
                pa = (ha + 1e-10) / (ha.sum() + n_bins * 1e-10)
                pb = (hb + 1e-10) / (hb.sum() + n_bins * 1e-10)
                kl_vals.append(_kl(pa, pb))
            except Exception:
                continue

        if kl_vals:
            thresholds[feature] = float(np.percentile(kl_vals, percentile))
        else:
            thresholds[feature] = 0.0

    return thresholds


def compute_kl_divergences(
    reference_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], n_bins: int = 10
) -> dict[str, float]:
    out: dict[str, float] = {}
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


def check_drift(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    thresholds: dict[str, float],
    logger: logging.Logger | None = None,
) -> tuple[bool, dict[str, dict]]:
    log = logger or logging.getLogger(__name__)
    kl_vals = compute_kl_divergences(reference_df, test_df, features)
    results: dict[str, dict] = {}
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
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p + eps
    q = q + eps
    return float(np.sum(p * np.log(p / q)))


def _hist_probs(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins)
    probs = counts.astype(float) / counts.sum() if counts.sum() > 0 else np.zeros_like(counts, dtype=float)
    return probs


def bootstrap_kl_threshold(
    series: pd.Series,
    n_boot: int = 500,
    sample_frac: float = 0.5,
    bins: int | str | None = "auto",
) -> float:
    """Estimate an empirical KL threshold by bootstrap-resampling the training series.

    Returns the 95th percentile of KL(subsample_a || subsample_b) computed
    across `n_boot` trials using identical binning on the full series.
    """
    arr = series.dropna().to_numpy()
    if arr.size == 0:
        return 0.0

    # choose bin edges on full distribution
    bin_edges = np.histogram_bin_edges(arr, bins=bins)
    kls = []
    n = max(1, int(len(arr) * sample_frac))
    rng = np.random.default_rng(seed=0)
    for _ in range(n_boot):
        a = rng.choice(arr, size=n, replace=True)
        b = rng.choice(arr, size=n, replace=True)
        pa = _hist_probs(a, bins=bin_edges)
        pb = _hist_probs(b, bins=bin_edges)
        kl = _kl_divergence(pa, pb)
        kls.append(kl)
    return float(np.percentile(kls, 95))


def compute_kl_between(series_ref: pd.Series, series_new: pd.Series, bins: int | str | None = "auto") -> float:
    arr_ref = series_ref.dropna().to_numpy()
    arr_new = series_new.dropna().to_numpy()
    if arr_ref.size == 0 or arr_new.size == 0:
        return 0.0
    bin_edges = np.histogram_bin_edges(arr_ref, bins=bins)
    p = _hist_probs(arr_ref, bins=bin_edges)
    q = _hist_probs(arr_new, bins=bin_edges)
    return _kl_divergence(p, q)


def detect_drift(
    df_ref: pd.DataFrame,
    df_new: pd.DataFrame,
    features: Iterable[str],
    n_boot: int = 500,
    sample_frac: float = 0.5,
    bins: int | str | None = "auto",
) -> dict[str, dict]:
    """Compute per-feature KL and compare against bootstrap thresholds.

    Returns a mapping: feature -> {"kl": float, "threshold": float, "drift": bool}
    """
    results: dict[str, dict] = {}
    for f in features:
        if f not in df_ref.columns or f not in df_new.columns:
            results[f] = {"kl": 0.0, "threshold": 0.0, "drift": False}
            continue
        thr = bootstrap_kl_threshold(df_ref[f], n_boot=n_boot, sample_frac=sample_frac, bins=bins)
        kl = compute_kl_between(df_ref[f], df_new[f], bins=bins)
        results[f] = {"kl": kl, "threshold": thr, "drift": kl > thr}
    return results
"""
monitoring/drift_check.py — Feature distribution drift detection.

Detects distribution drift in production data vs. training reference using
KL divergence on key numeric features. The threshold is **empirically derived**
from training data bootstrapping, not a fixed constant.

Bootstrap Threshold Derivation (Data-Sourced Expectation)
----------------------------------------------------------
We resample the training data ~1000 times into random 50/50 subsamples,
computing KL(subsample_A || subsample_B) for each feature. The 95th
percentile of these bootstrap KL values becomes our "natural noise floor"—
the drift amount we expect from random sampling variation alone.

**Why bootstrap?**
  - KL divergence is unbounded and scale-dependent (no universal threshold).
  - By deriving the threshold from the data itself, we anchor it to the
    actual feature distributions and binning scheme.
  - The 95th percentile is a defensible choice: it allows genuine natural
    variation while catching systematic shifts.

**Feature Selection:**
  - Restricted to top 3–4 **numeric** features by model feature importance.
  - Rationale: (1) Feature importances in sklearn pipelines are indexed by
    post-transform columns (post-OneHotEncoder), so categorical importances
    don't map 1:1 back to original features; (2) KL divergence is naturally
    suited for continuous distributions (what we bin and compute); (3) top
    features ensure we monitor the most impactful variables.
  - Example top features: lead_time, adr, stays_in_week_nights (from RF
    feature_importances_).

**Binning Strategy:**
  - Quantile-based (10 bins per feature) ensures equal-count bins and
    robustness to outliers. Avoids extreme bin widths that plague fixed-width
    binning on skewed data.

References
----------
- KL divergence: D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
  (asymmetric; measures how Q diverges from P's perspective)
- Quantile binning: pd.qcut() ensures each bin has ~equal count.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path


def estimate_kl_threshold(
    reference_df: pd.DataFrame,
    features: list[str],
    n_bootstrap: int = 1000,
    percentile: float = 95.0,
    n_bins: int = 10,
    logger: logging.Logger | None = None,
) -> dict[str, float]:
    """
    Derive empirical KL divergence thresholds from bootstrap resampling.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        Training data used to establish the baseline distribution.
    features : list[str]
        Feature names to compute thresholds for.
    n_bootstrap : int
        Number of bootstrap resamples (default 1000).
    percentile : float
        Percentile of bootstrap KL values to use as threshold (default 95).
    n_bins : int
        Number of quantile bins for density estimation (default 10).
    logger : logging.Logger | None
        Logger for progress messages.
    
    Returns
    -------
    dict[str, float]
        Mapping of {feature_name: kl_threshold}
    """
    log = logger or logging.getLogger(__name__)
    
    log.info(
        f"Estimating KL thresholds via bootstrap (n={n_bootstrap}, "
        f"percentile={percentile}, n_bins={n_bins}) ..."
    )
    
    thresholds = {}
    rng = np.random.default_rng(seed=42)  # Reproducible
    
    for feature in features:
        kl_values = []
        
        for _ in range(n_bootstrap):
            # Resample into two independent 50/50 subsamples
            idx_a = rng.choice(len(reference_df), size=len(reference_df) // 2, replace=False)
            idx_b = np.setdiff1d(np.arange(len(reference_df)), idx_a)
            
            sample_a = reference_df.iloc[idx_a][feature].dropna()
            sample_b = reference_df.iloc[idx_b][feature].dropna()
            
            # Bin both samples using the same quantile boundaries
            try:
                # Use quantiles from combined data to ensure stable bins
                combined = pd.concat([sample_a, sample_b])
                quantiles = np.linspace(0, 1, n_bins + 1)
                bin_edges = combined.quantile(quantiles).unique()
                
                if len(bin_edges) < 2:
                    continue  # Skip if insufficient unique values
                
                # Compute density (count / total) for each bin
                hist_a, _ = np.histogram(sample_a, bins=bin_edges)
                hist_b, _ = np.histogram(sample_b, bins=bin_edges)
                
                # Normalize to probabilities; add small smoothing to avoid log(0)
                p_a = (hist_a + 1e-10) / (hist_a.sum() + n_bins * 1e-10)
                p_b = (hist_b + 1e-10) / (hist_b.sum() + n_bins * 1e-10)
                
                # KL(p_a || p_b) — how much b diverges from a
                kl = np.sum(p_a * (np.log(p_a) - np.log(p_b)))
                kl_values.append(kl)
            except Exception as e:
                log.warning(f"Bootstrap iteration failed for {feature}: {e}")
                continue
        
        if kl_values:
            threshold = np.percentile(kl_values, percentile)
            thresholds[feature] = threshold
            log.info(f"  {feature}: KL threshold = {threshold:.6f} (p{percentile})")
        else:
            log.warning(f"No valid bootstrap KL values for {feature}; skipping")
    
    return thresholds


def compute_kl_divergence(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute KL divergence between reference and test distributions.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline training distribution.
    test_df : pd.DataFrame
        Test/production distribution to check for drift.
    features : list[str]
        Feature names to compute divergence for.
    n_bins : int
        Number of quantile bins.
    
    Returns
    -------
    dict[str, float]
        Mapping of {feature_name: kl_divergence}
    """
    kl_divergences = {}
    
    for feature in features:
        ref = reference_df[feature].dropna()
        test = test_df[feature].dropna()
        
        try:
            # Use quantiles from reference to compute bins
            quantiles = np.linspace(0, 1, n_bins + 1)
            bin_edges = ref.quantile(quantiles).unique()
            
            if len(bin_edges) < 2:
                continue
            
            hist_ref, _ = np.histogram(ref, bins=bin_edges)
            hist_test, _ = np.histogram(test, bins=bin_edges)
            
            p_ref = (hist_ref + 1e-10) / (hist_ref.sum() + n_bins * 1e-10)
            p_test = (hist_test + 1e-10) / (hist_test.sum() + n_bins * 1e-10)
            
            kl = np.sum(p_ref * (np.log(p_ref) - np.log(p_test)))
            kl_divergences[feature] = kl
        except Exception as e:
            logging.warning(f"KL computation failed for {feature}: {e}")
    
    return kl_divergences


def check_drift(
    reference_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    thresholds: dict[str, float],
    logger: logging.Logger | None = None,
) -> tuple[bool, dict[str, dict]]:
    """
    Check for feature distribution drift against established thresholds.
    
    Parameters
    ----------
    reference_df : pd.DataFrame
        Baseline distribution (training data).
    test_df : pd.DataFrame
        Test/production data to check.
    features : list[str]
        Feature names to check.
    thresholds : dict[str, float]
        KL thresholds (e.g., from estimate_kl_threshold).
    logger : logging.Logger | None
        Logger for output.
    
    Returns
    -------
    tuple[bool, dict]
        (no_drift_detected, results_dict)
        - no_drift_detected: True if all features pass
        - results_dict: {feature: {kl, threshold, passed}}
    """
    log = logger or logging.getLogger(__name__)
    
    kl_values = compute_kl_divergence(reference_df, test_df, features)
    
    results = {}
    all_pass = True
    
    for feature in features:
        if feature not in kl_values:
            results[feature] = {"status": "skipped", "reason": "KL computation failed"}
            continue
        
        kl = kl_values[feature]
        threshold = thresholds.get(feature, float("inf"))
        passed = kl <= threshold
        
        results[feature] = {
            "kl_divergence": kl,
            "threshold": threshold,
            "passed": passed,
        }
        
        status = "✓ PASS" if passed else "✗ DRIFT"
        log.info(f"{status} {feature}: KL={kl:.6f} (threshold={threshold:.6f})")
        
        if not passed:
            all_pass = False
    
    return all_pass, results
