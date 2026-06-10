"""
Download the Hotel Booking Demand CSV and write `data/hotel_bookings.parquet`.

Sources (first match wins):
  1. Environment variable HOTEL_BOOKINGS_CSV — path to a local CSV (e.g. from Kaggle).
  2. Default public mirror of the same dataset (TidyTuesday export of N. Antonia et al.).

The upstream CSV includes ADR outliers above $1,000 and a market_segment value
"Undefined" that is outside the training data contract. By default this script
writes a modeling extract that keeps only rows satisfying that contract so
pre-training Great Expectations checks pass on the parquet artifact.
Set RAW_PARQUET=1 to serialize the full table without those filters.

The default row filters mirror the Great Expectations contract in
``tests/test_data_quality.py`` (ADR bounds and closed ``market_segment`` set)
so that ``python scripts/serialize_data.py`` followed by
``python tests/test_data_quality.py`` succeeds without hand-editing data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

# Same contract as tests/test_data_quality.py
ALLOWED_MARKET_SEGMENTS = {
    "Direct",
    "Corporate",
    "Online TA",
    "Offline TA/TO",
    "Groups",
    "Aviation",
    "Complementary",
}

DEFAULT_CSV_URL = (
    "https://raw.githubusercontent.com/rfordatascience/tidytuesday/"
    "master/data/2020/2020-02-11/hotels.csv"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_csv() -> pd.DataFrame:
    local = os.environ.get("HOTEL_BOOKINGS_CSV")
    if local:
        path = Path(local).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"HOTEL_BOOKINGS_CSV not found: {path}")
        return pd.read_csv(path)

    tmp = project_root() / "data" / "_download_hotels.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("HOTEL_BOOKINGS_CSV_URL", DEFAULT_CSV_URL)
    print(f"Downloading CSV from {url}", file=sys.stderr)
    urlretrieve(url, tmp)
    df = pd.read_csv(tmp)
    tmp.unlink(missing_ok=True)
    return df


def apply_training_contract(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.copy()
    out = out[out["lead_time"].notna()]
    out = out[(out["adr"] >= 0) & (out["adr"] <= 1000)]
    out = out[out["market_segment"].isin(ALLOWED_MARKET_SEGMENTS)]
    dropped = before - len(out)
    if dropped:
        print(
            f"Dropped {dropped} rows outside training contract "
            f"(ADR 0-1000, allowed market_segment).",
            file=sys.stderr,
        )
    return out


def main() -> None:
    root = project_root()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "hotel_bookings.parquet"

    df = load_csv()
    if os.environ.get("RAW_PARQUET", "").strip() in {"1", "true", "yes"}:
        out = df
        print("Writing raw parquet (RAW_PARQUET set).", file=sys.stderr)
    else:
        out = apply_training_contract(df)

    # Add deterministic row_id for reproducible hash-based splitting in A/B tests
    # Uses positional index (after filtering) to ensure stability across re-serialization
    out = out.reset_index(drop=True).copy()
    out.insert(0, "row_id", out.index.astype(str))

    out.to_parquet(parquet_path, index=False)
    print(f"Wrote {len(out)} rows to {parquet_path} (with row_id column)")


if __name__ == "__main__":
    main()
