from __future__ import annotations

from pathlib import Path

from riboshift.config import Config
from riboshift.io_utils import read_table


REQUIRED_CORE_TABLES = {
    "source_records": [
        "ingest_row_id",
        "source_database",
        "task_family",
        "evidence_tier",
        "label_origin",
        "canonical_smiles",
        "rna_sequence",
    ],
    "rna_table": ["rna_id", "rna_sequence", "rna_cluster_id"],
    "ligand_table": ["ligand_id", "canonical_smiles", "inchikey", "murcko_scaffold"],
    "provenance_records": ["evidence_id", "source_database", "evidence_tier", "label_origin"],
    "interaction_records": ["interaction_id", "evidence_id", "rna_id", "ligand_id", "interaction_label", "task_family"],
    "view_annotations": ["analysis_view", "interaction_id"],
}


def validate_config(config: Config) -> list[str]:
    errors: list[str] = []
    total = config.split.train + config.split.valid + config.split.test
    if abs(total - 1.0) > 1e-6:
        errors.append("Split ratios must sum to 1.0")
    if config.min_eval_rows_per_fold < 1:
        errors.append("min_eval_rows_per_fold must be at least 1")
    if config.min_eval_direct_decoy_groups_per_fold < 0:
        errors.append("min_eval_direct_decoy_groups_per_fold must be non-negative")
    for source in config.sources:
        if not source.enabled:
            continue
        if source.mode == "materialized_table" and source.table_path is None:
            errors.append(f"{source.name}: materialized_table source requires table_path")
        if source.mode == "builder_adapter" and source.builder_path is None:
            errors.append(f"{source.name}: builder_adapter source requires builder_path")
    return errors


def validate_artifact_dir(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir) / "tables"
    errors: list[str] = []
    # Validation stays schema-level on purpose; release checks should be quick.
    for name, required_cols in REQUIRED_CORE_TABLES.items():
        path = root / f"{name}.parquet"
        if not path.exists():
            errors.append(f"Missing artifact table: {path}")
            continue
        frame = read_table(path)
        missing = [col for col in required_cols if col not in frame.columns]
        if missing:
            errors.append(f"{name}: missing columns {missing}")
    split_path = root / "split_assignments.parquet"
    decoy_path = root / "matched_decoy_candidates.parquet"
    if split_path.exists():
        split_frame = read_table(split_path)
        required = ["interaction_id", "analysis_view", "split_name", "fold_role"]
        missing = [col for col in required if col not in split_frame.columns]
        if missing:
            errors.append(f"split_assignments: missing columns {missing}")
    if decoy_path.exists():
        decoy_frame = read_table(decoy_path)
        required = ["anchor_interaction_id", "positive_ligand_id", "decoy_ligand_id", "decoy_regime"]
        missing = [col for col in required if col not in decoy_frame.columns]
        if missing:
            errors.append(f"matched_decoy_candidates: missing columns {missing}")
    return errors
