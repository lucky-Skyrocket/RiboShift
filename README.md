# RiboShift

RiboShift is a benchmark for RNA-small molecule interaction modeling under biological and chemical distribution shifts. This repository contains the normalized benchmark tables, fixed split assignments, matched-candidate pools, audit summaries, and construction code used in the manuscript.

## Release snapshot

- Release version: `RiboShift v1.0`
- Release type: immutable benchmark snapshot
- Primary benchmark view: `direct_only`
- Primary evaluation regime: matched-decoy target-conditional ranking
- Companion metadata file: `riboshift_croissant.json`

## Repository layout

- `code/riboshift/`: core package for ingestion, normalization, validation, and split generation
- `experiments/`: utilities for split audits, decoy-regime export, and result aggregation
- `data/tables/`: released benchmark tables
- `data/metadata/`: dataset metadata, benchmark statistics, source summaries, split summaries, and checksums
- `data/schemas/`: schemas for the released tables
- `configs/`: example configuration files
- `report/`: supplementary code notes for benchmark construction

## Data Files

The data files include normalized RNA and ligand tables, provenance-linked interaction records, view annotations, fixed split assignments, matched-decoy candidates, benchmark summaries, schemas, and dataset metadata.

## Documentation

The documentation records how the benchmark was built and how the released files should be interpreted:

- `README.md`: overview, repository layout, intended use, licensing, and archival plan
- `data/metadata/dataset_metadata.json`: machine-readable dataset summary, responsible-use fields, versioning policy, and redistribution scope
- `data/metadata/*.json`: benchmark statistics, source-ingestion summaries, split-quality summaries, and checksums
- `data/schemas/*.json`: table schemas for the released benchmark tables
- `report/1_benchmark_construction.md`: benchmark-construction notes
- `CHANGELOG.md`: release identifier and change policy for future benchmark versions
- `LICENSE_NOTICE.md`: licensing structure for benchmark artifacts and upstream source constraints

## Installation

```bash
pip install -e .
```

## Construction Commands

The `riboshift` command runs ingestion, validation, core table construction, and split generation.

```bash
riboshift ingest --config configs/benchmark_config.yaml
riboshift validate --config configs/benchmark_config.yaml
riboshift build-core --config configs/benchmark_config.yaml
riboshift make-splits --config configs/benchmark_config.yaml
```

## Experiment Utilities

The `experiments/` directory contains the training adapters, evaluation utilities, audit scripts, and result aggregation code used for the reported experiments.

Primary benchmark scripts:

- Primary benchmark baselines:
  - `train_bindti_baseline.py`
  - `train_deeprsma_baseline.py`
  - `train_graphdta_baseline.py`
  - `train_rnasmol_baseline.py`
  - `train_smrtnet_baseline.py`
- Reference and comparative baselines:
  - `train_char_baseline.py`
  - `train_descriptor_baseline.py` (includes descriptor-based logistic-regression, random-forest, and k-nearest-neighbor baselines)
  - `train_retrieval_baseline.py`
  - `train_deeprna_baseline.py`
- Transfer-learning analysis:
  - `transfer_char_baseline.py`
- Orchestration and audit helpers:
  - `run_suite.py`
  - `aggregate_results.py`
  - `build_split_audit.py`
  - `export_decoy_regimes.py`
  - `common.py`

These scripts provide the RiboShift-side adapters and aggregation code used in the experiments. They do not vendor full third-party baseline repositories; where needed, they call locally obtained upstream code and supply the benchmark-specific integration layer.

## Supplementary code notes

Benchmark-construction routines are summarized in [report/1_benchmark_construction.md](report/1_benchmark_construction.md).

## Versioning

RiboShift releases are versioned snapshots. Source additions, normalization-rule changes, split-policy changes, and audit/export updates require a new benchmark version.

## Licensing and provenance

Newly created RiboShift benchmark artifacts, including normalized benchmark tables, split assignments, matched-decoy candidates, schemas, summaries, metadata, and benchmark-specific documentation, are released under the Creative Commons Attribution 4.0 International license (CC BY 4.0).

Upstream source datasets, tables, and database records remain subject to their original licenses, terms of use, and citation requirements. The CC BY 4.0 license applies to the newly created RiboShift release artifacts and should not be interpreted as relicensing upstream source resources. Provenance records and source-level summaries are retained in the release package to support attribution and downstream inspection.

## Long-term hosting

After de-anonymization, each public benchmark release will be archived in a DOI-backed repository and linked from the public project page.
