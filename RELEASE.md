# v0.2.3

## v0.2.3 - 2026-04-23

### Changed
- Bump openfilter SDK, align CI workflow with shared release gate (source-paths)

- Fix release workflow secret names: `PYPI_API_TOKEN` → `PLAINSIGHT_PYPI_TOKEN`, `DOCKERHUB_TOKEN` → `DOCKERHUB_ACCESS_TOKEN` (org-level secret names). Without this the PyPI / Docker Hub tokens resolved to empty and no package has been published since the migration.
- Bump openfilter dependency to `>=0.1.30`.

# Changelog
ChatTag filter release notes
- Bump openfilter to 1.0.0

## [Unreleased]

## v0.2.2 - 2026-04-20

### Changed
- Remove redundant ci.yaml (shared workflow handles PR testing)
- Add push + pull_request triggers to create-release.yaml


## v0.2.1 - 2026-04-14

### Changed
- Add CI/CD workflows: create-release.yaml (Docker Hub publishing), ci.yaml (PR testing), security-scan.yaml (Grype)
- Bump openfilter dependency to >=0.1.27
- Extend Python support to 3.13
- Update docker-compose.yaml image tags to 0.1.27
- Update Makefile IMAGE to Docker Hub path


## v0.2.0 - 2026-04-02

### Added
- Standardized OpenFilter output: `schema_version` (`"1.0"`) on stream metadata (`meta.chatgpt_annotator`) and on each `labels.jsonl` line for downstream pipeline integration.
- Documented output contract in `docs/output_contract.md` and JSON Schema in `schemas/chattag_output.schema.json`.

### Changed
- Annotation payloads are classification-only: each label is `present` and `confidence`; extra fields from the model are not persisted.
- Shutdown: binary datasets from `labels.jsonl`; multilabel COCO export (`multilabel_datasets/`) when `output_schema` has more than one label (full-image boxes per active label).

## v0.1.2 - 2025-10-09

### Added
- Added automatic multilabel COCO dataset generation with full-image bounding boxes when bbox schema is present.

## v0.1.1 - 2025-09-29

### Changed
- Updated documentation

## v0.1.0 - 2025-02-22

### Added
- Initial Release: new ChatTag filter
