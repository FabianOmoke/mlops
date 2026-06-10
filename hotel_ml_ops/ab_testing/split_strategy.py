from __future__ import annotations

import hashlib
from typing import Iterable


def assign_variant(row_id: str | int, experiment_id: str) -> int:
    key = f"{row_id}:{experiment_id}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    return int.from_bytes(h, "big") % 2


def assign_variants(row_ids: Iterable[str], experiment_id: str) -> list[int]:
    return [assign_variant(r, experiment_id) for r in row_ids]
