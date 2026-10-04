from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

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


def load_rfam_mapping(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    frame = pd.read_csv(file_path)
    columns = {c.lower(): c for c in frame.columns}
    seq_col = columns.get("rna_sequence") or columns.get("sequence")
    rfam_col = columns.get("rfam_id")
    if not seq_col or not rfam_col:
        return {}
    mapping = {}
    for row in frame[[seq_col, rfam_col]].dropna().itertuples(index=False):
        sequence, rfam_id = row
        seq, valid = normalize_sequence(sequence)
        if valid and rfam_id:
            mapping[seq] = str(rfam_id)
    return mapping


def _similar_enough(seq_a: str, seq_b: str, threshold: float) -> bool:
    len_a = len(seq_a)
    len_b = len(seq_b)
    if not len_a or not len_b:
        return False
    if abs(len_a - len_b) / max(len_a, len_b) > (1.0 - threshold):
        return False
    if seq_a[:4] == seq_b[:4] or seq_a[-4:] == seq_b[-4:]:
        return SequenceMatcher(None, seq_a, seq_b).ratio() >= threshold
    kmers_a = {seq_a[i : i + 4] for i in range(max(1, len_a - 3))}
    kmers_b = {seq_b[i : i + 4] for i in range(max(1, len_b - 3))}
    if not kmers_a.intersection(kmers_b):
        return False
    return SequenceMatcher(None, seq_a, seq_b).ratio() >= threshold


def cluster_sequences(sequences: list[str], threshold: float = 0.9) -> dict[str, str]:
    unique_sequences = []
    seen = set()
    for seq in sequences:
        if seq and seq not in seen:
            seen.add(seq)
            unique_sequences.append(seq)
    unique_sequences.sort(key=lambda item: (-len(item), item))
    clusters: list[tuple[str, list[str]]] = []
    mapping: dict[str, str] = {}
    for seq in unique_sequences:
        assigned = None
        # Use a simple greedy pass here; the dataset is modest enough that this
        # remains easy to inspect and debug.
        for cluster_index, (representative, members) in enumerate(clusters, start=1):
            if _similar_enough(seq, representative, threshold):
                assigned = f"cluster90:{cluster_index:05d}"
                members.append(seq)
                break
        if assigned is None:
            clusters.append((seq, [seq]))
            assigned = f"cluster90:{len(clusters):05d}"
        mapping[seq] = assigned
    return mapping


def assign_group_ids(
    sequences: list[str],
    rfam_map: dict[str, str],
    threshold: float,
) -> tuple[dict[str, str], dict[str, str]]:
    missing = [seq for seq in sequences if seq and seq not in rfam_map]
    cluster_map = cluster_sequences(missing, threshold=threshold)
    group_map: dict[str, str] = {}
    group_source: dict[str, str] = {}
    for seq in sequences:
        if seq in rfam_map:
            group_map[seq] = f"rfam:{rfam_map[seq]}"
            group_source[seq] = "rfam"
        elif seq in cluster_map:
            group_map[seq] = cluster_map[seq]
            group_source[seq] = "cluster90"
        else:
            group_map[seq] = "cluster90:00000"
            group_source[seq] = "cluster90"
    return group_map, group_source
