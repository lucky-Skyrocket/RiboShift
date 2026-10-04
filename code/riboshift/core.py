from __future__ import annotations

from pathlib import Path

import pandas as pd

from riboshift.chem_utils import descriptor_dict, standardize_smiles
from riboshift.config import Config
from riboshift.ingest import run_ingest
from riboshift.io_utils import ensure_dir, write_json, write_table
from riboshift.rna_utils import assign_group_ids, load_rfam_mapping, normalize_sequence, normalize_structure


def _infer_rna_type(source_database: str, task_family: str) -> str:
    if source_database == "RNALigands":
        return "motif"
    if source_database == "HARIBOSS":
        return "structure_fragment"
    if task_family == "legacy_task_binary":
        return "task_sequence"
    return "unknown"


def _prepare_records(ingested: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    invalid_sequence = 0
    invalid_smiles = 0
    invalid_structure = 0
    for row in ingested.itertuples(index=False):
        # Sequence and ligand normalization are the hard filters here; structure
        # is retained when possible but not required for every source.
        normalized_sequence, is_valid_sequence = normalize_sequence(row.rna_sequence)
        normalized_smiles = standardize_smiles(row.canonical_smiles or row.ligand_repr_raw)
        if not is_valid_sequence:
            invalid_sequence += 1
            continue
        if not normalized_smiles:
            invalid_smiles += 1
            continue
        normalized_structure = normalize_structure(row.rna_structure_raw, len(normalized_sequence))
        if row.rna_structure_raw and not normalized_structure:
            invalid_structure += 1
        rows.append(
            {
                **row._asdict(),
                "rna_sequence": normalized_sequence,
                "rna_structure": normalized_structure,
                "canonical_smiles": normalized_smiles,
            }
        )
    prepared = pd.DataFrame(rows)
    audit = {
        "rows_in": int(len(ingested)),
        "rows_kept": int(len(prepared)),
        "invalid_sequence": int(invalid_sequence),
        "invalid_smiles": int(invalid_smiles),
        "invalid_structure": int(invalid_structure),
    }
    return prepared, audit


def _build_rna_entities(prepared: pd.DataFrame, config: Config) -> pd.DataFrame:
    rfam_map = load_rfam_mapping(config.rfam_mapping_path)
    unique_sequences = prepared["rna_sequence"].drop_duplicates().tolist()
    group_map, grouping_source = assign_group_ids(unique_sequences, rfam_map, threshold=config.rna_similarity_threshold)
    rows = []
    for idx, sequence in enumerate(unique_sequences, start=1):
        subset = prepared.loc[prepared["rna_sequence"] == sequence]
        structure = subset["rna_structure"].value_counts().index[0] if subset["rna_structure"].astype(str).str.len().gt(0).any() else ""
        source_database = subset.iloc[0]["source_database"]
        task_family = subset.iloc[0]["task_family"]
        rfam_id = rfam_map.get(sequence, "")
        rows.append(
            {
                "rna_id": f"RNA{idx:06d}",
                "rna_sequence": sequence,
                "rna_structure": structure,
                "rna_type": _infer_rna_type(source_database, task_family),
                "rfam_id": rfam_id,
                "rna_cluster_id": group_map[sequence],
                "grouping_source": grouping_source[sequence],
                "sequence_length": int(len(sequence)),
            }
        )
    return pd.DataFrame(rows)


def _build_ligand_entities(prepared: pd.DataFrame) -> pd.DataFrame:
    unique_smiles = sorted(prepared["canonical_smiles"].drop_duplicates().tolist())
    rows = []
    for idx, smiles in enumerate(unique_smiles, start=1):
        descriptor_payload = descriptor_dict(smiles)
        rows.append({"ligand_id": f"LIG{idx:06d}", **descriptor_payload})
    return pd.DataFrame(rows)


def _build_evidence_and_interactions(
    prepared: pd.DataFrame,
    rna_entities: pd.DataFrame,
    ligand_entities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rna_map = dict(zip(rna_entities["rna_sequence"], rna_entities["rna_id"]))
    ligand_map = dict(zip(ligand_entities["canonical_smiles"], ligand_entities["ligand_id"]))
    evidence_rows = []
    interaction_rows = []
    for idx, row in enumerate(prepared.itertuples(index=False), start=1):
        evidence_id = f"EVID{idx:07d}"
        interaction_id = f"INT{idx:07d}"
        evidence_rows.append(
            {
                "evidence_id": evidence_id,
                "source_database": row.source_database,
                "source_mode": row.source_mode,
                "source_asset_path": row.source_asset_path,
                "source_manifest_path": row.source_manifest_path,
                "source_record_id": row.source_record_id,
                "task_family": row.task_family,
                "evidence_tier": row.evidence_tier,
                "label_origin": row.label_origin,
                "reference": row.reference,
                "year": row.year,
            }
        )
        interaction_rows.append(
            {
                "interaction_id": interaction_id,
                "evidence_id": evidence_id,
                "rna_id": rna_map[row.rna_sequence],
                "ligand_id": ligand_map[row.canonical_smiles],
                "interaction_label": int(row.label_raw),
                "support_level": row.evidence_tier,
                "task_family": row.task_family,
                "evidence_tier": row.evidence_tier,
                "label_origin": row.label_origin,
                "is_legacy_binary_source": bool(row.task_family == "legacy_task_binary"),
                "source_database": row.source_database,
            }
        )
    return pd.DataFrame(evidence_rows), pd.DataFrame(interaction_rows)


def _build_analysis_views(interactions_main: pd.DataFrame) -> pd.DataFrame:
    direct = interactions_main.loc[interactions_main["task_family"] == "direct_interaction", ["interaction_id"]].copy()
    direct["analysis_view"] = "direct_only"
    all_sources = interactions_main[["interaction_id"]].copy()
    all_sources["analysis_view"] = "all_sources"
    return pd.concat([direct, all_sources], ignore_index=True)[["analysis_view", "interaction_id"]]


def _view_stats(
    interactions_main: pd.DataFrame,
    analysis_views: pd.DataFrame,
    rna_entities: pd.DataFrame,
    ligand_entities: pd.DataFrame,
) -> dict:
    global_entity_stats = {
        "rna_entities": int(len(rna_entities)),
        "ligand_entities": int(len(ligand_entities)),
        "interactions": int(len(interactions_main)),
        "sources": {str(k): int(v) for k, v in interactions_main["source_database"].value_counts().sort_index().items()},
    }
    stats = {"global_entity_stats": global_entity_stats}
    for view_name in ["direct_only", "all_sources"]:
        ids = set(analysis_views.loc[analysis_views["analysis_view"] == view_name, "interaction_id"].tolist())
        subset = interactions_main.loc[interactions_main["interaction_id"].isin(ids)].copy()
        key = f"{view_name}_benchmark_stats"
        stats[key] = {
            "interactions": int(len(subset)),
            "positives": int((subset["interaction_label"] == 1).sum()),
            "negatives": int((subset["interaction_label"] == 0).sum()),
            "task_family": {str(k): int(v) for k, v in subset["task_family"].value_counts().sort_index().items()},
            "sources": {str(k): int(v) for k, v in subset["source_database"].value_counts().sort_index().items()},
            "legacy_binary_rows": int(subset["is_legacy_binary_source"].sum()),
        }
    return stats


def build_core(config: Config, output_dir: str | Path) -> dict:
    output_dir = ensure_dir(output_dir)
    ingested, ingest_audit = run_ingest(config, output_dir)
    prepared, prep_audit = _prepare_records(ingested)
    if prepared.empty:
        raise RuntimeError("All ingested rows were filtered out during normalization")
    # print(f"[debug] prepared rows={len(prepared)} unique_rna={prepared['rna_sequence'].nunique()}")
    rna_entities = _build_rna_entities(prepared, config)
    ligand_entities = _build_ligand_entities(prepared)
    evidence_records, interactions_main = _build_evidence_and_interactions(prepared, rna_entities, ligand_entities)
    analysis_views = _build_analysis_views(interactions_main)
    stats = _view_stats(interactions_main, analysis_views, rna_entities, ligand_entities)
    write_table("rna_table", rna_entities, output_dir)
    write_table("ligand_table", ligand_entities, output_dir)
    write_table("provenance_records", evidence_records, output_dir)
    write_table("interaction_records", interactions_main, output_dir)
    write_table("view_annotations", analysis_views, output_dir)
    audit = {
        "ingest": ingest_audit,
        "prepare": prep_audit,
        "stats": stats,
    }
    write_json(Path(output_dir) / "metadata" / "data_build_summary.json", audit)
    write_json(Path(output_dir) / "metadata" / "benchmark_statistics.json", stats)
    return audit
