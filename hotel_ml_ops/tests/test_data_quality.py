"""
Pre-training data quality checks for `data/hotel_bookings.parquet`.

Expectations below are chosen for a demand / cancellation modeling use case on
the Hotel Booking Demand dataset (Antonio, Almeida, and Nunes; same lineage as
the Kaggle release). They encode a **data contract** between the analytics
warehouse export and downstream training: violations should block training and
trigger pipeline or schema review.

**Completeness (lead_time)**

In a production PMS or central reservations system, every confirmed booking has
a creation timestamp and an arrival date. ``lead_time`` is derived from those
facts. A null ``lead_time`` means the export is incomplete, keys do not line up
with operational reality, or ETL corrupted dates—so the row is unusable for any
time-based or demand feature and must not enter the model silently.

**ADR distribution ($0–$1,000 inclusive)**

ADR (average daily rate) cannot be negative (accounting / sign errors). The upper
bound is a **business guardrail for this hotel mix** (city + resort properties in
the study): rates far above typical published BAR for these segments are almost
always data errors, currency mistakes, or ultra-rare packages that would dominate
loss and distort learned "normal" price–behavior relationships. The numeric cap
was chosen as a **historical revenue-ops benchmark** aligned with the bulk of
observed stays after exploratory analysis (raw exports in this lineage include a
handful of extreme ADR spikes into the thousands). The contract matches what the
model is allowed to see at training time.

**Market segment (closed category set)**

``market_segment`` is treated as an enumerated channel field maintained by
revenue management (Direct, TA/TO, Groups, etc.). The allowed set is the **source
of truth** for model inputs: if a new code appears (e.g. a future "Meta-search"
bucket), the trained model has no calibrated behavior for that channel, so the
check **must fail** to force relabeling, retraining, or an explicit schema change
rather than silent extrapolation.

**Mixed attributes (non-trivial dataset)**

The serialized Parquet retains the full table schema from the source CSV: in
addition to numeric metrics (e.g. ``lead_time``, ``adr``) and categorical codes
(``market_segment``), textual / string attributes remain available for future
features—for example ``reservation_status_date`` (date string from the PMS),
``country`` (ISO-style guest country codes as text), ``meal``, and identifier-like
text fields such as ``agent`` / ``company`` where populated.
"""

from __future__ import annotations

from pathlib import Path

import great_expectations as ge
import pandas as pd

# Project root = parent of ``tests/``; Parquet is always under ``data/`` here so
# reviewers can run ``python tests/test_data_quality.py`` without manual path
# setup or changing the working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = PROJECT_ROOT / "data" / "hotel_bookings.parquet"

# Examples of numeric, categorical, and textual columns preserved in the Parquet
# (see module docstring). Used by ``test_mixed_attribute_columns_present``.
_NUMERIC_EXAMPLE = "lead_time"
_CATEGORICAL_EXAMPLE = "market_segment"
_TEXTUAL_EXAMPLES = ("reservation_status_date", "country", "meal", "agent", "company")


def _load_frame() -> pd.DataFrame:
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PARQUET_PATH}. Run: python scripts/serialize_data.py "
            "(from the project root, after pip install -r requirements.txt)."
        )
    return pd.read_parquet(PARQUET_PATH)


def test_row_count_non_trivial():
    """Assignment scale check: modeling extract must exceed 20k rows."""
    df = _load_frame()
    assert len(df) > 20_000, f"Expected >20,000 rows for a non-trivial extract, got {len(df)}"


def test_mixed_attribute_columns_present():
    """
    Numeric, categorical, and textual attributes coexist in the columnar file.

    Textual examples include PMS-style dates and dimensions stored as strings
    (``reservation_status_date``), guest origin (``country``), meal plan codes,
    and optional ``agent`` / ``company`` identifiers.
    """
    df = _load_frame()
    assert _NUMERIC_EXAMPLE in df.columns
    assert pd.api.types.is_numeric_dtype(df[_NUMERIC_EXAMPLE]), (
        f"{_NUMERIC_EXAMPLE} should be numeric for demand features"
    )

    assert _CATEGORICAL_EXAMPLE in df.columns
    seg = df[_CATEGORICAL_EXAMPLE]
    assert seg.dtype == object or pd.api.types.is_string_dtype(seg), (
        "market_segment should be string-like categorical storage"
    )

    text_hits = [
        c
        for c in _TEXTUAL_EXAMPLES
        if c in df.columns and pd.api.types.is_string_dtype(df[c])
    ]
    assert text_hits, (
        f"Expected at least one string column among {_TEXTUAL_EXAMPLES}, "
        f"got dtypes sample: {df.dtypes.head(20).to_dict()}"
    )


def test_hotel_data():
    # 1. Load the serialized columnar data (path resolved from this file — no cwd assumptions).
    df = ge.from_pandas(_load_frame())

    # 2. Null Value Test (Requirement 1)
    # Reasoning: Lead time is a primary key-adjacent metric for demand forecasting.
    res_null = df.expect_column_values_to_not_be_null("lead_time")
    assert res_null["success"], "Lead time contains null values!"

    # 3. Distribution Test: ADR (Requirement 2)
    # Reasoning: ADR must be positive and within realistic luxury bounds.
    res_adr = df.expect_column_values_to_be_between("adr", min_value=0, max_value=1000)
    assert res_adr["success"], "ADR distribution is outside acceptable business bounds."

    # 4. Distribution Test: Market Segment (Requirement 2)
    # Reasoning: Ensures the model only receives known business channels.
    allowed_segments = [
        "Direct",
        "Corporate",
        "Online TA",
        "Offline TA/TO",
        "Groups",
        "Aviation",
        "Complementary",
    ]
    res_seg = df.expect_column_values_to_be_in_set("market_segment", allowed_segments)
    assert res_seg["success"], "Unrecognized market segment detected!"


if __name__ == "__main__":
    test_row_count_non_trivial()
    test_mixed_attribute_columns_present()
    test_hotel_data()
    print("All pre-training tests passed!")
