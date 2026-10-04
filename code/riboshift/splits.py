from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random

import numpy as np
import pandas as pd

from riboshift.config import Config
from riboshift.io_utils import ensure_dir, read_table, write_json, write_table


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


def _allocate_units(unit_to_rows: dict[str, list[int]], row_task_family: dict[int, str], config: Config) -> dict[str, str]:
    if len(unit_to_rows) < 3:
        raise RuntimeError("Split infeasible: fewer than 3 unique units")
    total_rows = sum(len(rows) for rows in unit_to_rows.values())
    targets = _ratio_targets(total_rows, config)
    family_totals = defaultdict(int)
    for rows in unit_to_rows.values():
        for row_id in rows:
            family_totals[row_task_family[row_id]] += 1
    role_family_targets = {
        role: {
            family: int(round(count * targets[role] / total_rows))
            for family, count in family_totals.items()
        }
        for role in targets
    }
    assigned_rows = {role: 0 for role in targets}
    assigned_families = {role: defaultdict(int) for role in targets}
    unit_roles: dict[str, str] = {}
    sorted_units = sorted(unit_to_rows.items(), key=lambda item: len(item[1]), reverse=True)

    # Seed one unit per role so grouped splits never collapse to fewer than
    # train/valid/test, then greedily fill the remaining deficit.
    seed_roles = ["train", "valid", "test"]
    seeded_units = sorted_units[: len(seed_roles)]
    remaining_units = sorted_units[len(seed_roles) :]
    for role, (unit_id, rows) in zip(seed_roles, seeded_units):
        unit_roles[unit_id] = role
        assigned_rows[role] += len(rows)
        for row_id in rows:
            assigned_families[role][row_task_family[row_id]] += 1

    for unit_id, rows in remaining_units:
        unit_family = defaultdict(int)
        for row_id in rows:
            unit_family[row_task_family[row_id]] += 1
        deficits = {role: targets[role] - assigned_rows[role] for role in ["train", "valid", "test"]}
        positive_deficit_roles = [role for role, deficit in deficits.items() if deficit > 0]
        if positive_deficit_roles:
            best_role = max(
                positive_deficit_roles,
                key=lambda role: (
                    deficits[role],
                    sum(role_family_targets[role][family] - assigned_families[role][family] for family in unit_family),
                ),
            )
        else:
            best_role = min(
                ["train", "valid", "test"],
                key=lambda role: (
                    assigned_rows[role] + len(rows) - targets[role],
                    assigned_rows[role],
                ),
            )
        unit_roles[unit_id] = best_role
        assigned_rows[best_role] += len(rows)
        for family, count in unit_family.items():
            assigned_families[best_role][family] += count
    return unit_roles


def _group_assign(frame: pd.DataFrame, unit_col: str, config: Config) -> pd.Series:
    unit_to_rows = defaultdict(list)
    row_task_family = {}
    for row in frame.itertuples(index=True):
        unit_to_rows[str(getattr(row, unit_col))].append(row.Index)
        row_task_family[row.Index] = row.task_family
    unit_roles = _allocate_units(unit_to_rows, row_task_family, config)
    assignments = {row_index: unit_roles[str(unit)] for unit, rows in unit_to_rows.items() for row_index in rows}
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
    component_to_rows = defaultdict(list)
    row_task_family = {}
    for row in frame.itertuples(index=True):
        component = find(f"rna::{row.rna_group_id}")
        component_to_rows[component].append(row.Index)
        row_task_family[row.Index] = row.task_family
    component_roles = _allocate_units(component_to_rows, row_task_family, config)
    assignments = {row_index: component_roles[component] for component, rows in component_to_rows.items() for row_index in rows}
    return pd.Series(assignments).sort_index()


def _ensure_disjoint(frame: pd.DataFrame, assignments: pd.Series, key_col: str) -> None:
    role_to_keys = {}
    for role in ["train", "valid", "test"]:
        role_to_keys[role] = set(frame.loc[assignments == role, key_col].astype(str))
    if role_to_keys["train"] & role_to_keys["valid"]:
        raise RuntimeError(f"Split infeasible: overlap detected for {key_col} between train and valid")
    if role_to_keys["train"] & role_to_keys["test"]:
        raise RuntimeError(f"Split infeasible: overlap detected for {key_col} between train and test")
    if role_to_keys["valid"] & role_to_keys["test"]:
        raise RuntimeError(f"Split infeasible: overlap detected for {key_col} between valid and test")


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


def _build_ligand_context(ligand_entities: pd.DataFrame) -> dict:
    descriptor_cols = ["mol_wt", "logp", "hbd", "hba", "tpsa", "ring_count"]
    ligand_ids = ligand_entities["ligand_id"].astype(str).tolist()
    ligand_group_by_id = {
        str(row.ligand_id): str(row.murcko_scaffold or "")
        for row in ligand_entities[["ligand_id", "murcko_scaffold"]].itertuples(index=False)
    }
    descriptors = ligand_entities[descriptor_cols].fillna(0.0).astype(float).to_numpy()
    if len(descriptors) == 0:
        return {
            "ligand_ids": [],
            "ligand_index": {},
            "ligand_group_by_id": {},
            "distance_matrix": np.zeros((0, 0), dtype=float),
            "sorted_neighbors": np.zeros((0, 0), dtype=int),
        }
    descriptor_mean = descriptors.mean(axis=0)
    descriptor_std = descriptors.std(axis=0)
    descriptor_std = np.where(descriptor_std > 1e-8, descriptor_std, 1.0)
    standardized = (descriptors - descriptor_mean) / descriptor_std
    diff = standardized[:, None, :] - standardized[None, :, :]
    distance_matrix = np.sum(diff * diff, axis=2)
    sorted_neighbors = np.argsort(distance_matrix, axis=1)
    ligand_index = {ligand_id: idx for idx, ligand_id in enumerate(ligand_ids)}
    return {
        "ligand_ids": ligand_ids,
        "ligand_index": ligand_index,
        "ligand_group_by_id": ligand_group_by_id,
        "distance_matrix": distance_matrix,
        "sorted_neighbors": sorted_neighbors,
    }


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
    artifact_dir = ensure_dir(artifact_dir)
    table_dir = Path(artifact_dir) / "tables"
    interactions = read_table(table_dir / "interaction_records.parquet")
    analysis_views = read_table(table_dir / "view_annotations.parquet")
    rna_entities = read_table(table_dir / "rna_table.parquet")
    ligand_entities = read_table(table_dir / "ligand_table.parquet")
    interactions = interactions.merge(rna_entities[["rna_id", "rna_cluster_id"]], on="rna_id", how="left")
    interactions = interactions.merge(ligand_entities[["ligand_id", "murcko_scaffold"]], on="ligand_id", how="left")
    interactions = interactions.rename(columns={"rna_cluster_id": "rna_group_id", "murcko_scaffold": "ligand_group_id"})
    ligand_context = _build_ligand_context(ligand_entities)
    split_rows = []
    decoy_frames = []
    split_audit = {}
    for view_name in ["direct_only", "all_sources"]:
        # We materialize each benchmark view separately so the release can keep
        # direct-only evaluation distinct from the mixed-source view.
        ids = set(analysis_views.loc[analysis_views["analysis_view"] == view_name, "interaction_id"].tolist())
        view_frame = interactions.loc[interactions["interaction_id"].isin(ids)].copy().reset_index(drop=True)
        split_audit[view_name] = {}
        random_assignments = _random_assign(
            view_frame,
            config,
            ["task_family", "interaction_label"] if view_name == "all_sources" else None,
        )
        family_assignments = _group_assign(view_frame, "rna_group_id", config)
        scaffold_assignments = _group_assign(view_frame, "ligand_group_id", config)
        dual_assignments = _dual_cold_assign(view_frame, config)
        assignment_payload = {
            "random": random_assignments,
            "family_shift": family_assignments,
            "scaffold_shift": scaffold_assignments,
            "dual_cold": dual_assignments,
        }
        for split_name, assignment_series in assignment_payload.items():
            role_series = assignment_series.reindex(view_frame.index)
            if role_series.isna().any():
                raise RuntimeError(f"Split assignment missing rows for {view_name}/{split_name}")
            if split_name in {"family_shift", "dual_cold"}:
                _ensure_disjoint(view_frame, role_series, "rna_group_id")
            if split_name in {"scaffold_shift", "dual_cold"}:
                _ensure_disjoint(view_frame, role_series, "ligand_group_id")
            annotated = view_frame.copy()
            annotated["fold_role"] = role_series.values
            annotated["analysis_view"] = view_name
            annotated["split_name"] = split_name
            annotated["seed"] = config.seed
            direct_subset = annotated.loc[annotated["task_family"] == "direct_interaction"].copy()
            feasibility_issues = _assess_split_feasibility(annotated, direct_subset, config)
            if feasibility_issues:
                split_audit[view_name][split_name] = {
                    "status": "unsupported",
                    "rows": int(len(annotated)),
                    "fold_counts": {str(k): int(v) for k, v in annotated["fold_role"].value_counts().sort_index().items()},
                    "task_family": {str(k): int(v) for k, v in annotated["task_family"].value_counts().sort_index().items()},
                    "decoys": 0,
                    "issues": feasibility_issues,
                }
                continue
            split_rows.append(
                annotated[
                    [
                        "interaction_id",
                        "analysis_view",
                        "split_name",
                        "fold_role",
                        "seed",
                        "rna_group_id",
                        "ligand_group_id",
                        "task_family",
                        "interaction_label",
                    ]
                ]
            )
            decoys = _generate_decoys(view_name, split_name, annotated, direct_subset, ligand_context, config)
            if not decoys.empty:
                decoy_frames.append(decoys)
            split_audit[view_name][split_name] = {
                "status": "ok",
                "rows": int(len(annotated)),
                "fold_counts": {str(k): int(v) for k, v in annotated["fold_role"].value_counts().sort_index().items()},
                "task_family": {str(k): int(v) for k, v in annotated["task_family"].value_counts().sort_index().items()},
                "decoys": int(len(decoys)),
            }
            # print(f"[debug] {view_name}/{split_name}: rows={len(annotated)} decoys={len(decoys)}")
    split_assignments = (
        pd.concat(split_rows, ignore_index=True)
        if split_rows
        else pd.DataFrame(
            columns=[
                "interaction_id",
                "analysis_view",
                "split_name",
                "fold_role",
                "seed",
                "rna_group_id",
                "ligand_group_id",
                "task_family",
                "interaction_label",
            ]
        )
    )
    decoy_candidates = (
        pd.concat(decoy_frames, ignore_index=True)
        if decoy_frames
        else pd.DataFrame(
            columns=[
                "analysis_view",
                "split_name",
                "anchor_interaction_id",
                "rna_id",
                "positive_ligand_id",
                "decoy_ligand_id",
                "fold_role",
                "decoy_regime",
                "match_distance",
            ]
        )
    )
    write_table("split_assignments", split_assignments, artifact_dir)
    write_table("matched_decoy_candidates", decoy_candidates, artifact_dir)
    write_json(Path(artifact_dir) / "metadata" / "split_quality_summary.json", split_audit)
    return split_audit
