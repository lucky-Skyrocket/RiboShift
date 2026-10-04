from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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


def _resolve_path(value: str | None, asset_root: Path, config_dir: Path) -> Path | None:
    if not value:
        return None
    raw = Path(value)
    if raw.is_absolute():
        return raw
    # Prefer the declared asset root, but still allow config-local paths.
    asset_candidate = asset_root / raw
    if asset_candidate.exists() or str(value).startswith("_tmp"):
        return asset_candidate
    return config_dir / raw


def load_config(path: str | Path) -> Config:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    asset_root = Path(payload.get("asset_root", ".")).expanduser()
    if not asset_root.is_absolute():
        asset_root = (config_path.parent / asset_root).resolve()
    split_payload = payload.get("split", {}) or {}
    split = SplitConfig(
        train=float(split_payload.get("train", 0.8)),
        valid=float(split_payload.get("valid", 0.1)),
        test=float(split_payload.get("test", 0.1)),
    )
    sources: list[SourceConfig] = []
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
    cfg = Config(
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
    return cfg


def config_to_dict(config: Config) -> dict[str, Any]:
    return {
        "asset_root": str(config.asset_root),
        "seed": config.seed,
        "rna_similarity_threshold": config.rna_similarity_threshold,
        "rfam_mapping_path": str(config.rfam_mapping_path) if config.rfam_mapping_path else None,
        "decoy_k": config.decoy_k,
        "min_eval_rows_per_fold": config.min_eval_rows_per_fold,
        "min_eval_direct_decoy_groups_per_fold": config.min_eval_direct_decoy_groups_per_fold,
        "split": {
            "train": config.split.train,
            "valid": config.split.valid,
            "test": config.split.test,
        },
        "sources": [
            {
                "name": src.name,
                "mode": src.mode,
                "enabled": src.enabled,
                "required": src.required,
                "task_family": src.task_family,
                "evidence_tier": src.evidence_tier,
                "positive_only": src.positive_only,
                "table_path": str(src.table_path) if src.table_path else None,
                "metadata_path": str(src.metadata_path) if src.metadata_path else None,
                "manifest_path": str(src.manifest_path) if src.manifest_path else None,
                "builder_type": src.builder_type,
                "builder_path": str(src.builder_path) if src.builder_path else None,
                "download_url": src.download_url,
                "pmcid": src.pmcid,
                "file_format": src.file_format,
                "delimiter": src.delimiter,
                "has_header": src.has_header,
                "sheet_name": src.sheet_name,
                "column_map": src.column_map,
                "reference": src.reference,
                "year": src.year,
            }
            for src in config.sources
        ],
    }
