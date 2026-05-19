# MLOps Assignment — Task 2: Pre-Deployment Tests

## Overview

This task builds on the dataset and data quality tests from Task 1 ( Hotel Booking Demand dataset )
and implements a full pre-deployment ML pipeline. The goal is to simulate a realistic production
ML lifecycle: orchestrated steps, a trained and versioned model artifact, robustness validation,
and graceful error handling.

---

## Dataset

**Hotel Booking Demand** — 119,386 bookings from two Portuguese hotels (city + resort),
sourced from Antonio, Almeida & Nunes (2019) via the TidyTuesday GitHub mirror.

- **Target variable:** `is_canceled` (binary: 0 = kept, 1 = canceled)
- **Input features:** mix of numeric (e.g. `lead_time`, `adr`), categorical
  (e.g. `market_segment`, `deposit_type`), and textual/string attributes
  (e.g. `country`, `meal`, `reservation_status_date`)
- **Cancellation rate:** ~37% — class imbalance handled via `class_weight="balanced"`

---

## Pipeline Orchestration

I used **Prefect (local runner)** as the orchestration framework because it requires
no external server for local execution, has clean task/flow abstractions, and produces
structured logs that make step failures easy to diagnose.

The pipeline is defined in `flow.py` and consists of three sequential steps:

### Step 1 — Data Quality Tests (`data_quality_step`)
Runs the Great Expectations checks from Task 1 directly as a Prefect task:
- Row count > 20,000
- Mixed attribute types present (numeric, categorical, textual)
- `lead_time` has no null values
- `adr` is within [0, 1000]
- `market_segment` only contains known business categories

If any check fails, the pipeline stops here and training is blocked.

### Step 2 — Model Training (`training_step`)
Trains a **RandomForestClassifier** (200 estimators) on 80% of the dataset (~95k rows),
wrapped in a full sklearn `Pipeline` with:
- `SimpleImputer` + `StandardScaler` for numeric features
- `SimpleImputer` + `OneHotEncoder` for categorical features

**Why RandomForest?** It handles mixed feature types well, is robust to skewed
distributions, requires minimal hyperparameter tuning for a strong baseline, and
produces reliable probability estimates for AUC evaluation.

**Why joblib over pickle?** joblib is the serialization format explicitly recommended
by scikit-learn for persisting estimators — it handles numpy arrays more efficiently
and produces smaller files.

The output is a serialized `.joblib` artifact stored under `models/` and registered
in the local model registry.

### Step 3 — Robustness Validation (`robustness_step`)
Loads the just-registered model from the registry and runs two robustness checks:

**Check 1 — ROC-AUC ≥ 0.80**
The model must achieve at least 0.80 AUC on the held-out test split (20%, ~24k rows).
This threshold was chosen because:
- A majority-class dummy classifier scores ~0.50
- Published baselines for this dataset report 0.87–0.93 AUC
- 0.80 is tight enough to catch under-fit or buggy models, while allowing
  for intentionally simplified feature sets

Result: **0.918 AUC** ✓

**Check 2 — Prediction flip rate ≤ 5% under Gaussian noise (σ=0.01)**
Adding small Gaussian noise to all numeric features must not change more than
5% of predictions. This simulates minor floating-point rounding or imputation
artifacts that a production system might introduce. A model that is highly
sensitive to such noise is likely over-fit or has an unstable decision boundary.
σ=0.01 corresponds to roughly 1% relative noise on a StandardScaler-normalized
feature — below any realistic measurement uncertainty in the source data.

Result: **0.23% flip rate** ✓

---

## Model Versioning

I implemented a lightweight custom model registry (`versioning/model_registry.py`)
rather than using MLflow, to keep the pipeline fully self-contained with no
external services. The registry persists all model metadata to `models/registry.json`
and supports three operations:

- **`list()`** — returns all registered models with their artifact paths and metadata
- **`load(model_id)`** — deserializes and returns the sklearn Pipeline for a given id
- **`get_metadata(model_id)`** — returns the registry entry without loading the model

Each registry entry stores:
- Artifact path
- Registration timestamp
- ROC-AUC metric
- Input schema (numeric + categorical feature lists)
- Output schema (target name, type, possible values)
- Code dependencies
- Training and test set sizes

---

## Error Handling

I introduced an artificial small-dataset error to simulate a data pipeline failure
or an interrupted training run. The guard is in `training/train.py`:

- If the training slice contains **fewer than 1,000 rows**, a `ValueError` is raised
  immediately with a descriptive message explaining the likely cause
- This is triggered artificially by setting `FORCE_SMALL_DATASET=1`
- The serialization uses an **atomic write** (temp file → rename) so a mid-write
  crash never leaves a corrupt artifact in the models directory

The rationale: for a ~120k-row dataset, a 1k training slice represents less than
1% of available signal. The resulting model would be statistically indistinguishable
from random guessing on any meaningful evaluation set, and silently registering it
would be worse than failing loudly.

---

## Per-Step Dependencies

Each pipeline step has its own `requirements.txt` listing only the dependencies
it strictly needs:

| Step | File | Key dependencies |
|---|---|---|
| Data quality | `tests/requirements.txt` | great-expectations, pandas, pyarrow |
| Training | `training/requirements.txt` | scikit-learn, joblib, pandas, pyarrow, numpy |
| Validation | `validation/requirements.txt` | scikit-learn, joblib, pandas, pyarrow, numpy |

A combined `requirements_task2.txt` is also provided for convenience.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements_task2.txt

# 2. Serialize the dataset (downloads from source if not present)
python scripts/serialize_data.py

# 3. Run the full pipeline
python flow.py

# 4. Test error handling (artificial small dataset injection)
FORCE_SMALL_DATASET=1 python flow.py
```

---

## Expected Output (normal run)
