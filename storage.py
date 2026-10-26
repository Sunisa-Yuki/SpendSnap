"""Save receipts to a local JSON file so they survive refreshes and restarts."""

import json
import os
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).parent / "data" / "receipts.json"


class StorageError(Exception):
    """The saved receipts file exists but could not be read."""


def load_receipts(path: Path = DATA_FILE) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        receipts = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageError(f"Could not read saved receipts in {path}.") from exc
    if not isinstance(receipts, list):
        raise StorageError(f"Saved receipts in {path} are not a list.")
    return receipts


def save_receipts(receipts: list[dict[str, Any]], path: Path = DATA_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first so a crash mid-save can't wipe existing receipts.
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(receipts, indent=2), encoding="utf-8")
    os.replace(temp_path, path)
