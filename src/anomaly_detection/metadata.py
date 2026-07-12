"""Read and write deployment metadata."""

import json
from pathlib import Path
from typing import Any


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    """Serialize model metadata as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load metadata JSON and validate its root object."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Metadata JSON must contain an object at its root.")
    return data
