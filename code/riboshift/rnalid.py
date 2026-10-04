from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests
from docx import Document

from riboshift.config import SourceConfig
from riboshift.io_utils import ensure_dir


def ensure_rnalid_docx(source: SourceConfig) -> Path:
    if source.builder_path is None:
        raise FileNotFoundError("RNALID builder_path is not configured")
    if source.builder_path.exists() and source.builder_path.stat().st_size > 0:
        return source.builder_path
    if not source.download_url:
        raise FileNotFoundError(f"RNALID source file is missing: {source.builder_path}")
    ensure_dir(source.builder_path.parent)
    response = requests.get(
        source.download_url,
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    content = response.content
    if response.status_code == 200 and content[:2] == b"PK":
        source.builder_path.write_bytes(content)
        return source.builder_path
    preview = response.text[:240] if "text" in response.headers.get("content-type", "") else ""
    raise FileNotFoundError(
        "RNALID supplement could not be auto-downloaded. "
        f"Status={response.status_code}. Place the supplement manually at {source.builder_path}. "
        f"Preview={preview!r}"
    )


def _normalize_header(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _header_mapping(header_cells: list[str]) -> dict[int, str]:
    mapping = {}
    for idx, cell in enumerate(header_cells):
        normalized = _normalize_header(cell)
        if not normalized:
            continue
        if "smiles" in normalized:
            mapping[idx] = "SMILES"
        elif "sequence" in normalized and "rna" in normalized:
            mapping[idx] = "Sequence"
        elif normalized == "sequence":
            mapping[idx] = "Sequence"
        elif "structure" in normalized:
            mapping[idx] = "Structure"
        elif "doi" in normalized or "reference" in normalized or "pmid" in normalized:
            mapping[idx] = "reference"
        elif normalized in {"id", "no", "number"} or "record id" in normalized:
            mapping[idx] = "source_record_id"
        elif normalized == "year":
            mapping[idx] = "year"
        elif normalized == "ligand":
            mapping[idx] = "ligand_name"
    return mapping


def parse_rnalid_docx(path: str | Path) -> pd.DataFrame:
    document = Document(str(path))
    frames: list[pd.DataFrame] = []
    for table in document.tables:
        raw_rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
        if not raw_rows:
            continue
        header_index = None
        header_map = None
        for idx, row in enumerate(raw_rows[:5]):
            # RNALID supplementary tables are not fully standardized across
            # documents, so we scan a few candidate header rows.
            current_map = _header_mapping(row)
            if "Sequence" in current_map.values() and "SMILES" in current_map.values():
                header_index = idx
                header_map = current_map
                break
        if header_index is None or header_map is None:
            continue
        parsed_rows = []
        for row in raw_rows[header_index + 1 :]:
            if not any(cell.strip() for cell in row):
                continue
            record = {
                "source_record_id": "",
                "SMILES": "",
                "Sequence": "",
                "Structure": "",
                "reference": "",
                "year": pd.NA,
            }
            for cell_index, field_name in header_map.items():
                if cell_index < len(row):
                    record[field_name] = row[cell_index].strip()
            if not record["SMILES"] or not record["Sequence"]:
                continue
            record["Label"] = 1
            parsed_rows.append(record)
        if parsed_rows:
            frames.append(pd.DataFrame(parsed_rows))
    if not frames:
        raise ValueError(f"No RNALID interaction tables with SMILES and RNA sequences were found in {path}")
    frame = pd.concat(frames, ignore_index=True)
    frame["Structure"] = frame.get("Structure", "").fillna("")
    frame["reference"] = frame.get("reference", "").fillna("")
    if "source_record_id" not in frame.columns:
        frame["source_record_id"] = [f"rnalid_{idx+1}" for idx in range(len(frame))]
    else:
        frame["source_record_id"] = frame["source_record_id"].replace("", pd.NA)
        frame["source_record_id"] = frame["source_record_id"].fillna(pd.Series([f"rnalid_{idx+1}" for idx in range(len(frame))]))
    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    else:
        frame["year"] = pd.NA
    frame = frame[["source_record_id", "SMILES", "Sequence", "Structure", "Label", "reference", "year"]].drop_duplicates().reset_index(drop=True)
    return frame
