# Supplementary code notes for benchmark construction

This document summarizes the main Python components used to construct the RiboShift benchmark release. The excerpts below retain the original logic and parameterization while omitting routine file I/O and repeated helper code that is not needed for understanding the construction workflow.

## Configuration and source definitions

The release is driven by a YAML configuration that specifies source tables, benchmark-wide thresholds, and split ratios.

```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SplitConfig:
    train: float = 0.8
    valid: float = 0.1
    test: float = 0.1


@dataclass
class SourceConfig:
    name: str
    mode: str
    enabled: bool = True
    required: bool = True
    task_family: str = "direct_interaction"
    evidence_tier: str = "validated_interaction"
    positive_only: bool = False
    table_path: Path | None = None
    metadata_path: Path | None = None
    manifest_path: Path | None = None
    builder_type: str | None = None
    builder_path: Path | None = None
    download_url: str | None = None
    pmcid: str | None = None
    file_format: str | None = None
    delimiter: str = "\t"
    has_header: bool = False
    sheet_name: str | None = None
    column_map: dict[str, str] = field(default_factory=dict)
    reference: str = ""
    year: int | None = None


@dataclass
class Config:
    asset_root: Path
    seed: int = 42
    rna_similarity_threshold: float = 0.9
    rfam_mapping_path: Path | None = None
    decoy_k: int = 10
    min_eval_rows_per_fold: int = 1
    min_eval_direct_decoy_groups_per_fold: int = 0
    split: SplitConfig = field(default_factory=SplitConfig)
    sources: list[SourceConfig] = field(default_factory=list)
    config_path: Path | None = None


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    split_payload = payload.get("split", {}) or {}
    split = SplitConfig(
        train=float(split_payload.get("train", 0.8)),
        valid=float(split_payload.get("valid", 0.1)),
        test=float(split_payload.get("test", 0.1)),
    )

    sources = []
    for entry in payload.get("sources", []) or []:
        src = dict(entry)
        sources.append(
            SourceConfig(
                name=str(src["name"]),
                mode=str(src["mode"]),
                enabled=bool(src.get("enabled", True)),
                required=bool(src.get("required", True)),
                task_family=str(src.get("task_family", "direct_interaction")),
                evidence_tier=str(src.get("evidence_tier", "validated_interaction")),
                positive_only=bool(src.get("positive_only", False)),
                table_path=_resolve_path(src.get("table_path"), asset_root, config_path.parent),
                metadata_path=_resolve_path(src.get("metadata_path"), asset_root, config_path.parent),
                manifest_path=_resolve_path(src.get("manifest_path"), asset_root, config_path.parent),
                builder_type=src.get("builder_type"),
                builder_path=_resolve_path(src.get("builder_path"), asset_root, config_path.parent),
                download_url=src.get("download_url"),
                pmcid=src.get("pmcid"),
                file_format=src.get("file_format"),
                delimiter=str(src.get("delimiter", "\t")),
                has_header=bool(src.get("has_header", False)),
                sheet_name=src.get("sheet_name"),
                column_map=dict(src.get("column_map", {}) or {}),
                reference=str(src.get("reference", "")),
                year=int(src["year"]) if src.get("year") not in (None, "") else None,
            )
        )

    return Config(
        asset_root=asset_root,
        seed=int(payload.get("seed", 42)),
        rna_similarity_threshold=float(payload.get("rna_similarity_threshold", 0.9)),
        rfam_mapping_path=_resolve_path(payload.get("rfam_mapping_path"), asset_root, config_path.parent),
        decoy_k=int(payload.get("decoy_k", 10)),
        min_eval_rows_per_fold=int(payload.get("min_eval_rows_per_fold", 1)),
        min_eval_direct_decoy_groups_per_fold=int(payload.get("min_eval_direct_decoy_groups_per_fold", 0)),
        split=split,
        sources=sources,
        config_path=config_path,
    )
```

## Source ingestion and schema alignment

Raw source tables are normalized into a shared evidence-level schema before any sequence or ligand filtering is applied.

```python
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


def _prepare_base_frame(source: SourceConfig) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if source.mode == "materialized_table":
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

        if source.builder_type == "rnalid_pmc_docx":
            docx_path = ensure_rnalid_docx(source)
            frame = parse_rnalid_docx(docx_path)
            return frame.copy(), None

    raise ValueError(f"Unsupported source mode: {source.mode}")


def load_source(source: SourceConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, meta = _prepare_base_frame(source)
    # print(f"[debug] loading {source.name}: {len(frame)} rows before filtering")

    if source.positive_only:
        frame = frame.loc[
            pd.to_numeric(frame["Label"], errors="coerce").fillna(0).astype(int) == 1
        ].copy()
        label_origin = "source_positive_only"
    else:
        label_origin = "source_binary_label"

    standardized = pd.DataFrame(
        {
            "source_database": source.name,
            "source_mode": source.mode,
            "source_asset_path": str((source.table_path or source.builder_path or Path("")).resolve()),
            "source_manifest_path": str(source.manifest_path.resolve()) if source.manifest_path else "",
            "source_record_id": _pick_source_record_id(meta, frame.index.to_series()).astype(str),
            "task_family": source.task_family,
            "evidence_tier": source.evidence_tier,
            "label_origin": label_origin,
            "rna_repr_raw": frame["Sequence"].astype(str),
            "rna_sequence": frame["Sequence"].astype(str),
            "rna_structure_raw": frame["Structure"].astype(str),
            "ligand_repr_raw": frame["SMILES"].astype(str),
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
    }
    return standardized, audit
```

## Conservative normalization of RNA and ligand representations

Sequence and ligand normalization are treated as hard filters. Structure is retained when available, but incomplete structure strings do not block a record if the sequence and ligand remain valid.

```python
VALID_BASES = set("ACGUN")


def normalize_sequence(sequence: str) -> tuple[str, bool]:
    text = "".join(str(sequence or "").upper().split()).replace("T", "U")
    if not text:
        return "", False
    return text, set(text).issubset(VALID_BASES)


def normalize_structure(structure: str, sequence_length: int) -> str:
    text = "".join(str(structure or "").split())
    if not text:
        return ""
    if len(text) != sequence_length:
        return ""
    return text


@lru_cache(maxsize=50000)
def standardize_smiles(smiles: str) -> str:
    smiles = str(smiles or "").strip()
    if not smiles or smiles.lower() in {"nan", "none", "null"}:
        return ""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""

    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        pass

    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return ""


def descriptor_dict(smiles: str) -> dict[str, float | int | str]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {
            "canonical_smiles": "",
            "inchikey": "",
            "murcko_scaffold": "",
            "mol_wt": 0.0,
            "logp": 0.0,
            "hbd": 0,
            "hba": 0,
            "tpsa": 0.0,
            "ring_count": 0,
        }

    canonical = standardize_smiles(smiles)
    return {
        "canonical_smiles": canonical,
        "inchikey": inchikey_of(canonical),
        "murcko_scaffold": scaffold_of(canonical),
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
    }
```

## Construction of benchmark entities and interaction tables

After normalization, the pipeline constructs RNA entities, ligand entities, provenance records, pair-level interaction records, and the two benchmark analysis views.

```python
def _prepare_records(ingested: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    invalid_sequence = 0
    invalid_smiles = 0
    invalid_structure = 0

    for row in ingested.itertuples(index=False):
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
    group_map, grouping_source = assign_group_ids(
        unique_sequences,
        rfam_map,
        threshold=config.rna_similarity_threshold,
    )

    rows = []
    for idx, sequence in enumerate(unique_sequences, start=1):
        subset = prepared.loc[prepared["rna_sequence"] == sequence]
        structure = (
            subset["rna_structure"].value_counts().index[0]
            if subset["rna_structure"].astype(str).str.len().gt(0).any()
            else ""
        )
        rows.append(
            {
                "rna_id": f"RNA{idx:06d}",
                "rna_sequence": sequence,
                "rna_structure": structure,
                "rfam_id": rfam_map.get(sequence, ""),
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
        rows.append({"ligand_id": f"LIG{idx:06d}", **descriptor_dict(smiles)})
    return pd.DataFrame(rows)


def _build_analysis_views(interactions_main: pd.DataFrame) -> pd.DataFrame:
    direct = interactions_main.loc[
        interactions_main["task_family"] == "direct_interaction",
        ["interaction_id"],
    ].copy()
    direct["analysis_view"] = "direct_only"

    all_sources = interactions_main[["interaction_id"]].copy()
    all_sources["analysis_view"] = "all_sources"

    return pd.concat([direct, all_sources], ignore_index=True)[["analysis_view", "interaction_id"]]


def build_core(config: Config, output_dir: str | Path) -> dict:
    output_dir = ensure_dir(output_dir)
    ingested, ingest_audit = run_ingest(config, output_dir)
    prepared, prep_audit = _prepare_records(ingested)
    if prepared.empty:
        raise RuntimeError("All ingested rows were filtered out during normalization")

    # print(f"[debug] prepared rows={len(prepared)} unique_rna={prepared['rna_sequence'].nunique()}")

    rna_entities = _build_rna_entities(prepared, config)
    ligand_entities = _build_ligand_entities(prepared)
    evidence_records, interactions_main = _build_evidence_and_interactions(
        prepared,
        rna_entities,
        ligand_entities,
    )
    analysis_views = _build_analysis_views(interactions_main)

    write_table("rna_table", rna_entities, output_dir)
    write_table("ligand_table", ligand_entities, output_dir)
    write_table("provenance_records", evidence_records, output_dir)
    write_table("interaction_records", interactions_main, output_dir)
    write_table("view_annotations", analysis_views, output_dir)

    # ...
    return audit
```

## Split assignment under multiple novelty protocols

The benchmark defines four evaluation protocols: random split, family shift, scaffold shift, and dual-cold splitting over both RNA groups and ligand scaffolds.

```python
def _ratio_targets(size: int, config: Config) -> dict[str, int]:
    train = int(round(size * config.split.train))
    valid = int(round(size * config.split.valid))
    test = size - train - valid
    return {"train": train, "valid": valid, "test": test}


def _random_assign(frame: pd.DataFrame, config: Config, stratify_cols: list[str] | None = None) -> pd.Series:
    rng = random.Random(config.seed)

    if stratify_cols:
        grouped = defaultdict(list)
        for idx, row in frame[stratify_cols].astype(str).iterrows():
            grouped["|".join(row.tolist())].append(idx)

        assignments = {}
        for indexes in grouped.values():
            rng.shuffle(indexes)
            n = len(indexes)
            targets = _ratio_targets(n, config)
            for role, subset in (
                ("train", indexes[: targets["train"]]),
                ("valid", indexes[targets["train"] : targets["train"] + targets["valid"]]),
                ("test", indexes[targets["train"] + targets["valid"] :]),
            ):
                for idx in subset:
                    assignments[idx] = role
        return pd.Series(assignments).sort_index()

    indexes = list(frame.index)
    rng.shuffle(indexes)
    targets = _ratio_targets(len(indexes), config)
    assignments = {}
    for role, subset in (
        ("train", indexes[: targets["train"]]),
        ("valid", indexes[targets["train"] : targets["train"] + targets["valid"]]),
        ("test", indexes[targets["train"] + targets["valid"] :]),
    ):
        for idx in subset:
            assignments[idx] = role
    return pd.Series(assignments).sort_index()


def _group_assign(frame: pd.DataFrame, unit_col: str, config: Config) -> pd.Series:
    unit_to_rows = defaultdict(list)
    row_task_family = {}
    for row in frame.itertuples(index=True):
        unit_to_rows[str(getattr(row, unit_col))].append(row.Index)
        row_task_family[row.Index] = row.task_family

    unit_roles = _allocate_units(unit_to_rows, row_task_family, config)
    assignments = {
        row_index: unit_roles[str(unit)]
        for unit, rows in unit_to_rows.items()
        for row_index in rows
    }
    return pd.Series(assignments).sort_index()


def _dual_cold_assign(frame: pd.DataFrame, config: Config) -> pd.Series:
    parent = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in frame.itertuples(index=True):
        union(f"rna::{row.rna_group_id}", f"scaf::{row.ligand_group_id}")

    # ...
    return pd.Series(assignments).sort_index()
```

## Matched-decoy generation and release gating

Matched decoys are generated for direct-interaction positives in validation and test folds only. Split variants that do not satisfy the minimum support thresholds are marked unsupported and excluded from the primary benchmark release.

```python
def _generate_decoys(
    view_name: str,
    split_name: str,
    annotated_frame: pd.DataFrame,
    direct_subset: pd.DataFrame,
    ligand_context: dict,
    config: Config,
) -> pd.DataFrame:
    ligand_ids: list[str] = ligand_context["ligand_ids"]
    ligand_index: dict[str, int] = ligand_context["ligand_index"]
    ligand_group_by_id: dict[str, str] = ligand_context["ligand_group_by_id"]
    sorted_neighbors: np.ndarray = ligand_context["sorted_neighbors"]

    positives_by_rna = defaultdict(set)
    for row in annotated_frame.itertuples(index=False):
        if int(row.interaction_label) == 1:
            positives_by_rna[row.rna_id].add(row.ligand_id)

    allowed_ligands_by_fold: dict[str, set[str]] = {}
    if split_name in {"scaffold_shift", "dual_cold"}:
        for fold_role in ["valid", "test"]:
            allowed_groups = set(
                annotated_frame.loc[annotated_frame["fold_role"] == fold_role, "ligand_group_id"]
                .fillna("")
                .astype(str)
                .replace("", pd.NA)
                .dropna()
                .tolist()
            )
            allowed_ligands_by_fold[fold_role] = {
                ligand_id
                for ligand_id, group_id in ligand_group_by_id.items()
                if group_id and group_id in allowed_groups
            }

    rows = []
    for anchor in direct_subset.itertuples(index=False):
        if anchor.fold_role not in {"valid", "test"}:
            continue
        if int(anchor.interaction_label) != 1:
            continue

        anchor_idx = ligand_index.get(anchor.ligand_id)
        if anchor_idx is None:
            continue

        selected = 0
        positive_ligands = positives_by_rna[anchor.rna_id]
        allowed_ligands = allowed_ligands_by_fold.get(anchor.fold_role)

        for candidate_idx in sorted_neighbors[anchor_idx]:
            candidate_ligand_id = ligand_ids[int(candidate_idx)]
            if candidate_ligand_id == anchor.ligand_id:
                continue
            if candidate_ligand_id in positive_ligands:
                continue
            if allowed_ligands is not None and candidate_ligand_id not in allowed_ligands:
                continue

            distance = float(ligand_context["distance_matrix"][anchor_idx, int(candidate_idx)])
            rows.append(
                {
                    "analysis_view": view_name,
                    "split_name": split_name,
                    "anchor_interaction_id": anchor.interaction_id,
                    "rna_id": anchor.rna_id,
                    "positive_ligand_id": anchor.ligand_id,
                    "decoy_ligand_id": candidate_ligand_id,
                    "fold_role": anchor.fold_role,
                    "decoy_regime": "generated_matched_decoy",
                    "match_distance": distance,
                }
            )
            selected += 1
            if selected >= config.decoy_k:
                break

    return pd.DataFrame(rows)


def _assess_split_feasibility(
    annotated: pd.DataFrame,
    direct_subset: pd.DataFrame,
    config: Config,
) -> list[str]:
    issues: list[str] = []
    for fold_role in ["valid", "test"]:
        fold_rows = int((annotated["fold_role"] == fold_role).sum())
        if fold_rows < config.min_eval_rows_per_fold:
            issues.append(
                f"{fold_role} rows {fold_rows} below min_eval_rows_per_fold={config.min_eval_rows_per_fold}"
            )

        direct_groups = int(
            (
                (direct_subset["fold_role"] == fold_role)
                & (direct_subset["interaction_label"].astype(int) == 1)
            ).sum()
        )
        if direct_groups < config.min_eval_direct_decoy_groups_per_fold:
            issues.append(
                f"{fold_role} direct_decoy_groups {direct_groups} below "
                f"min_eval_direct_decoy_groups_per_fold={config.min_eval_direct_decoy_groups_per_fold}"
            )

    return issues


def make_splits(config: Config, artifact_dir: str | Path) -> dict:
    # ...
    for view_name in ["direct_only", "all_sources"]:
        ids = set(
            analysis_views.loc[
                analysis_views["analysis_view"] == view_name,
                "interaction_id",
            ].tolist()
        )
        view_frame = interactions.loc[
            interactions["interaction_id"].isin(ids)
        ].copy().reset_index(drop=True)

        assignment_payload = {
            "random": _random_assign(
                view_frame,
                config,
                ["task_family", "interaction_label"] if view_name == "all_sources" else None,
            ),
            "family_shift": _group_assign(view_frame, "rna_group_id", config),
            "scaffold_shift": _group_assign(view_frame, "ligand_group_id", config),
            "dual_cold": _dual_cold_assign(view_frame, config),
        }

        for split_name, assignment_series in assignment_payload.items():
            annotated = view_frame.copy()
            annotated["fold_role"] = assignment_series.reindex(view_frame.index).values
            annotated["analysis_view"] = view_name
            annotated["split_name"] = split_name
            annotated["seed"] = config.seed

            direct_subset = annotated.loc[annotated["task_family"] == "direct_interaction"].copy()
            feasibility_issues = _assess_split_feasibility(annotated, direct_subset, config)
            if feasibility_issues:
                split_audit[view_name][split_name] = {
                    "status": "unsupported",
                    "rows": int(len(annotated)),
                    "decoys": 0,
                    "issues": feasibility_issues,
                }
                continue

            decoys = _generate_decoys(
                view_name,
                split_name,
                annotated,
                direct_subset,
                ligand_context,
                config,
            )

            # print(f"[debug] {view_name}/{split_name}: rows={len(annotated)} decoys={len(decoys)}")

    # ...
    write_table("split_assignments", split_assignments, artifact_dir)
    write_table("matched_decoy_candidates", decoy_candidates, artifact_dir)
    write_json(Path(artifact_dir) / "metadata" / "split_quality_summary.json", split_audit)
    return split_audit
```

## Command-line interface

The package exposes a small command-line interface for ingestion, validation, core table construction, and split generation.

```python
import argparse
from pathlib import Path

from riboshift.config import load_config
from riboshift.core import build_core
from riboshift.ingest import run_ingest
from riboshift.splits import make_splits
from riboshift.validate import validate_artifact_dir, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RiboShift benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest source tables into a standardized artifact table")
    ingest_parser.add_argument("--config", required=True)
    ingest_parser.add_argument("--output-dir", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate config and/or generated artifacts")
    validate_parser.add_argument("--config", required=False)
    validate_parser.add_argument("--artifact-dir", required=False)

    core_parser = subparsers.add_parser("build-core", help="Build core benchmark tables from source configs")
    core_parser.add_argument("--config", required=True)
    core_parser.add_argument("--output-dir", required=True)

    split_parser = subparsers.add_parser("make-splits", help="Generate benchmark splits and matched decoys")
    split_parser.add_argument("--config", required=True)
    split_parser.add_argument("--artifact-dir", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        config = load_config(args.config)
        run_ingest(config, Path(args.output_dir))
        return 0

    if args.command == "validate":
        errors = []
        if args.config:
            errors.extend(validate_config(load_config(args.config)))
        if args.artifact_dir:
            errors.extend(validate_artifact_dir(args.artifact_dir))
        if errors:
            for error in errors:
                print(error)
            return 1
        print("Validation passed")
        return 0

    if args.command == "build-core":
        config = load_config(args.config)
        build_core(config, Path(args.output_dir))
        return 0

    if args.command == "make-splits":
        config = load_config(args.config)
        make_splits(config, Path(args.artifact_dir))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
```
