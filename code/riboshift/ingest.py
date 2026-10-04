from __future__ import annotations

from pathlib import Path

import pandas as pd

from riboshift.adapters import STANDARD_COLUMNS, load_source
from riboshift.config import Config
from riboshift.io_utils import ensure_dir, write_json, write_table


def run_ingest(config: Config, output_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    output_dir = ensure_dir(output_dir)
    records = []
    source_audits = []
    for source in config.sources:
        if not source.enabled:
            continue
        try:
            frame, audit = load_source(source)
        except FileNotFoundError:
            if source.required:
                raise
            # Optional sources can be absent without invalidating the release.
            audit = {
                "source": source.name,
                "status": "skipped_missing_optional_source",
            }
            source_audits.append(audit)
            continue
        frame["config_source_name"] = source.name
        records.append(frame)
        source_audits.append(audit)
    if not records:
        raise RuntimeError("No sources were ingested")
    ingested = pd.concat(records, ignore_index=True)
    ingested["ingest_row_id"] = [f"INGEST_{idx:07d}" for idx in range(len(ingested))]
    ordered = ["ingest_row_id", *STANDARD_COLUMNS, "config_source_name"]
    ingested = ingested[ordered].copy()
    write_table("source_records", ingested, output_dir)
    audit = {
        "asset_root": str(config.asset_root),
        "rows_total": int(len(ingested)),
        "sources": source_audits,
    }
    write_json(Path(output_dir) / "metadata" / "source_ingestion_summary.json", audit)
    return ingested, audit
