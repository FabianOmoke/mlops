from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FlowConfigRegistry:
    """Registry for tracking flow versions and their configurations."""

    def __init__(self, path: Path | str) -> None:
        """Initialize registry at specified path."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict[str, Any]:
        """Read registry from disk."""
        with open(self.path) as f:
            return json.load(f)

    def _write(self, data: dict[str, Any]) -> None:
        """Write registry to disk."""
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self, flow_version_id: str, config: dict[str, Any], model_id: str | None = None
    ) -> None:
        """Register a flow version with its config and resulting model ID."""
        data = self._read()
        data[flow_version_id] = {"config": config, "model_id": model_id}
        self._write(data)

    def get(self, flow_version_id: str) -> dict[str, Any] | None:
        """Retrieve flow version config and model ID."""
        data = self._read()
        return data.get(flow_version_id)

    def list(self) -> dict[str, Any]:
        """List all registered flow versions."""
        return self._read()
