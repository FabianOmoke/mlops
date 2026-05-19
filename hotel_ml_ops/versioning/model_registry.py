"""
versioning/model_registry.py — Lightweight local model registry.

Provides three core capabilities required by the assignment:
  a) list()  — list all registered models and their artifact locations.
  b) load()  — load a model by its id (deserialize + return the pipeline).
  c) Each model entry stores metadata: input/output schema, code dependencies,
     training metrics, and artifact path.

Storage
-------
All metadata is persisted as a single JSON file (models/registry.json).
This is intentionally simple — no database, no external service — so the
pipeline is fully locally executable as required.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


class ModelRegistry:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self._write([])

    def _read(self) -> list[dict]:
        with open(self.registry_path) as f:
            return json.load(f)

    def _write(self, entries: list[dict]) -> None:
        with open(self.registry_path, "w") as f:
            json.dump(entries, f, indent=2)

    def register(
        self,
        model_id: str,
        artifact_path: Path,
        metrics: dict[str, Any],
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        dependencies: list[str],
        **extra_metadata: Any,
    ) -> None:
        entries = self._read()
        entries.append({
            "model_id": model_id,
            "artifact_path": str(artifact_path),
            "registered_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "dependencies": dependencies,
            **extra_metadata,
        })
        self._write(entries)

    def list(self) -> list[dict]:
        return self._read()

    def load(self, model_id: str) -> Any:
        entries = self._read()
        matches = [e for e in entries if e["model_id"] == model_id]
        if not matches:
            raise KeyError(
                f"Model '{model_id}' not found in registry. "
                f"Available ids: {[e['model_id'] for e in entries]}"
            )
        artifact_path = Path(matches[-1]["artifact_path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Artifact for model '{model_id}' not found at {artifact_path}."
            )
        return joblib.load(artifact_path)

    def get_metadata(self, model_id: str) -> dict:
        entries = self._read()
        matches = [e for e in entries if e["model_id"] == model_id]
        if not matches:
            raise KeyError(f"Model '{model_id}' not found in registry.")
        return matches[-1]
