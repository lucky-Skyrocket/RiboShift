from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: dict | list) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_table(name: str, frame: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)
    table_dir = ensure_dir(out_dir / "tables")
    manifest_dir = ensure_dir(out_dir / "schemas")
    table_path = table_dir / f"{name}.parquet"
    frame.to_parquet(table_path, index=False)
    # Keep a small schema sidecar next to each released table.
    manifest = {
        "name": name,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "path": str(table_path),
    }
    write_json(manifest_dir / f"{name}.json", manifest)
    return table_path


def read_table(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
