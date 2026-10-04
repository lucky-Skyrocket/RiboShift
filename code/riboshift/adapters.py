from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from riboshift.config import SourceConfig
from riboshift.chem_utils import standardize_smiles
from riboshift.rnalid import ensure_rnalid_docx, parse_rnalid_docx

STANDARD_COLUMNS = [
    "source_database",
    "source_mode",
    "source_asset_path",
    "source_manifest_path",
    "source_record_id",
    "task_family",
    "evidence_tier",
    "label_origin",
    "rna_repr_raw",
    "rna_sequence",
    "rna_structure_raw",
    "ligand_repr_raw",
    "canonical_smiles",
    "label_raw",
    "reference",
    "year",
]


def _read_tabular(
    path: Path,
    file_format: str | None,
    delimiter: str,
    has_header: bool,
    sheet_name: str | None,
) -> pd.DataFrame:
    # Keep source readers minimal here; downstream normalization happens later.
    suffix = (file_format or path.suffix.lstrip(".")).lower()
    if suffix in {"tsv", "txt"}:
        if has_header:
            return pd.read_csv(path, sep=delimiter)
        return pd.read_csv(path, sep=delimiter, header=None)
    if suffix == "csv":
        if has_header:
            return pd.read_csv(path)
        return pd.read_csv(path, header=None)
    if suffix in {"xlsx", "xlsm", "xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    raise ValueError(f"Unsupported tabular format for {path}")


def _pick_source_record_id(meta: pd.DataFrame | None, index_series: pd.Series) -> pd.Series:
    if meta is None:
        return index_series.astype(str)
    for col in ["source_record_id", "source_id", "sample_id", "original_sample_id"]:
        if col in meta.columns:
            return meta[col].astype(str)
    if {"source_complex_id", "source_sample_key"}.issubset(meta.columns):
        return meta["source_complex_id"].astype(str) + ":" + meta["source_sample_key"].astype(str)
    return index_series.astype(str)


def _load_metadata(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _prepare_base_frame(source: SourceConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if source.mode == "materialized_table":
        if source.table_path is None or not source.table_path.exists():
            raise FileNotFoundError(f"Missing table_path for {source.name}: {source.table_path}")
        frame = _read_tabular(
            source.table_path,
            source.file_format,
            source.delimiter,
            source.has_header,
            source.sheet_name,
        )
        if list(frame.columns) == [0, 1, 2, 3]:
            frame = frame.rename(columns={0: "SMILES", 1: "Sequence", 2: "Structure", 3: "Label"})
        meta = _load_metadata(source.metadata_path)
        return frame, meta
    if source.mode == "builder_adapter":
        # Builder adapters cover source-specific tables that need a small amount
        # of parsing before they can enter the common schema.
        if source.builder_path is None or not source.builder_path.exists():
            raise FileNotFoundError(f"Missing builder_path for {source.name}: {source.builder_path}")
        if source.builder_type == "rbind_table":
            frame = _read_tabular(
                source.builder_path,
                source.file_format,
                source.delimiter,
                True,
                source.sheet_name,
            )
            rename_map = {
                "SMILES": "SMILES",
                "RNA sequence": "Sequence",
                "RNA structure": "Structure",
                "Label": "Label",
                "DOI": "reference",
                "No.": "source_record_id",
            }
            trimmed = frame.rename(columns=rename_map)
            keep = [
                col
                for col in ["SMILES", "Sequence", "Structure", "Label", "reference", "source_record_id"]
                if col in trimmed.columns
            ]
            return trimmed[keep].copy(), None
        if source.builder_type == "tabular":
            frame = _read_tabular(
                source.builder_path,
                source.file_format,
                source.delimiter,
                source.has_header,
                source.sheet_name,
            )
            if not source.has_header:
                columns = list(source.column_map.values())
                frame.columns = columns[: len(frame.columns)]
            reverse_map = {value: key for key, value in (source.column_map or {}).items()}
            frame = frame.rename(columns=reverse_map)
            rename_map = {
                "ligand": "SMILES",
                "sequence": "Sequence",
                "structure": "Structure",
                "label": "Label",
            }
            frame = frame.rename(columns=rename_map)
            return frame.copy(), None
        if source.builder_type == "rnalid_pmc_docx":
            docx_path = ensure_rnalid_docx(source)
            frame = parse_rnalid_docx(docx_path)
            return frame.copy(), None
        raise ValueError(f"Unsupported builder_type for {source.name}: {source.builder_type}")
    raise ValueError(f"Unsupported source mode: {source.mode}")


def load_source(source: SourceConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, meta = _prepare_base_frame(source)
    required = ["SMILES", "Sequence"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Source {source.name} is missing required columns: {missing}")
    if "Structure" not in frame.columns:
        frame["Structure"] = ""
    if "Label" not in frame.columns:
        frame["Label"] = 1

    if source.positive_only:
        frame = frame.loc[pd.to_numeric(frame["Label"], errors="coerce").fillna(0).astype(int) == 1].copy()
        label_origin = "source_positive_only"
    else:
        label_origin = "source_binary_label"

    frame = frame.reset_index(drop=True)
    meta = meta.reset_index(drop=True) if meta is not None else None
    source_record_id = _pick_source_record_id(meta, frame.index.to_series())

    if meta is not None and len(meta) == len(frame):
        sequence_raw = meta["Sequence"].astype(str) if "Sequence" in meta.columns else frame["Sequence"].astype(str)
        structure_raw = meta["Structure"].astype(str) if "Structure" in meta.columns else frame["Structure"].astype(str)
        ligand_raw = meta["SMILES"].astype(str) if "SMILES" in meta.columns else frame["SMILES"].astype(str)
    else:
        sequence_raw = frame["Sequence"].astype(str)
        structure_raw = frame["Structure"].astype(str)
        ligand_raw = frame["SMILES"].astype(str)

    # print(f"[debug] loaded source={source.name} rows={len(frame)}")

    standardized = pd.DataFrame(
        {
            "source_database": source.name,
            "source_mode": source.mode,
            "source_asset_path": str((source.table_path or source.builder_path or Path("")).resolve())
            if (source.table_path or source.builder_path)
            else "",
            "source_manifest_path": str(source.manifest_path.resolve()) if source.manifest_path else "",
            "source_record_id": source_record_id.astype(str),
            "task_family": source.task_family,
            "evidence_tier": source.evidence_tier,
            "label_origin": label_origin,
            "rna_repr_raw": sequence_raw.astype(str),
            "rna_sequence": frame["Sequence"].astype(str),
            "rna_structure_raw": structure_raw.astype(str),
            "ligand_repr_raw": ligand_raw.astype(str),
            "canonical_smiles": frame["SMILES"].astype(str).map(standardize_smiles),
            "label_raw": pd.to_numeric(frame["Label"], errors="coerce").fillna(0).astype(int),
            "reference": str(source.reference or ""),
            "year": source.year if source.year is not None else pd.NA,
        }
    )
    for column in STANDARD_COLUMNS:
        if column not in standardized.columns:
            standardized[column] = ""
    standardized = standardized[STANDARD_COLUMNS].copy()
    audit = {
        "source": source.name,
        "mode": source.mode,
        "rows_loaded": int(len(standardized)),
        "labels": {
            str(k): int(v)
            for k, v in standardized["label_raw"].value_counts().sort_index().items()
        },
        "positive_only": bool(source.positive_only),
        "task_family": source.task_family,
    }
    return standardized, audit
