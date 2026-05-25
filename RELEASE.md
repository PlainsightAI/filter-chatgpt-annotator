# v0.2.4

## v0.2.4 - 2026-05-25

### Fixed
- Security: mask `chatgpt_api_key` in the OpenFilter framework's post-normalize config log. The framework's built-in scrubber only hides URI-style `user:pass` credentials, so loose API keys were being emitted verbatim by `openfilter.filter_runtime.filter.Filter.__init__`. The key is now wrapped in a `SecretValue` holder at the tail of `normalize_config`, so framework `str(config)` paths render `***masked***`. Regression test covers `repr`/`str`/`format`. Operators running prior versions with a real key should rotate it.

### Changed
- Scripts: `filter_pet_classification.py` and `filter_multilabel.py` now accept `IMAGE_PATH` (auto-selecting `ImageIn` vs `VideoIn`), matching `filter_food_annotation.py`.
- Docs: drop references to example scripts that do not exist in this repo (`filter_medical_imaging.py`, `filter_industrial_quality.py`, `filter_simple_salad.py`, `filter_annotation.py`) from `README.md` and `scripts/README.md`.
- Bump openfilter to 1.1.0 (carried from the previously unreleased entry; merged in #12, first shipped here).

## v0.2.3 - 2026-04-23

### Changed
- Bump openfilter SDK, align CI workflow with shared release gate (source-paths)

- Fix release workflow secret names: `PYPI_API_TOKEN` → `PLAINSIGHT_PYPI_TOKEN`, `DOCKERHUB_TOKEN` → `DOCKERHUB_ACCESS_TOKEN` (org-level secret names). Without this the PyPI / Docker Hub tokens resolved to empty and no package has been published since the migration.
- Bump openfilter dependency to `>=0.1.30`.

# Changelog
ChatTag filter release notes

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
