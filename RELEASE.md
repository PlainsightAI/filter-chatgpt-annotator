# Changelog
ChatTag filter release notes

## [Unreleased]

## v0.3.3 - 2026-08-04

### Changed
- Update `openfilter[all]` to `>=1.2.1`.
- Dockerfile: prefer PyPI over the openfilter mirror as the primary pip index, so `publish-docker` no longer races against mirror sync immediately after `publish-to-pypi`.
- Grant `id-token: write` in `create-release.yaml` so the public release workflow can produce a keyless (cosign) SBOM attestation for the published image (once the shared SBOM steps land).
- Fix the `RELEASE.md` header (`# Changelog` first line; a stray `# v0.3.2` H1 plus a duplicated `# Changelog`/`[Unreleased]` block broke the changelog-parser).
- Pin the Docker base to `python:3.11.12-slim` (was `python:3.11-slim`).
- Point the `docker-compose.yaml` utility images at `containers.openfilter.io/plainsightai/openfilter-{video-in,webvis}:1.2.1` (were `:latest`).
- Update dev-tooling floors (`setuptools>=83.0.0`) and switch dev pins to range pins.

## v0.3.2 - 2026-07-26

### Changed
- `README.md`: use the exact HyperLabel™ wording specified by Shanker for the filter family.

## v0.3.1 - 2026-07-25

### Changed
- `README.md`: document this filter as part of the Plainsight **Hyperlabel** family of filters.

## v0.3.0 - 2026-05-25

### Added
- LangChain backend — `FilterChatTag` now talks to any LangChain-supported chat model (OpenAI, Google Gemini, Anthropic Claude, Ollama, …) via `init_chat_model`. All four providers ship in the default install.
- Structured output enforcement via `with_structured_output(Pydantic)` derived from `FILTER_OUTPUT_SCHEMA` — JSON shape is guaranteed by the provider's native mechanism (tool-calling / JSON-mode), no more `json.loads` fallback in the happy path.
- Migration guide ([`MIGRATION.md`](MIGRATION.md)) covering imports, env vars, metadata keys, and the architectural shift.

### Changed (breaking)
- Renamed import package, class, and Docker image:
  - `filter_chatgpt_annotator` → `filter_chattag` (Python import path)
  - `FilterChatgptAnnotator` / `FilterChatgptAnnotatorConfig` → `FilterChatTag` / `FilterChatTagConfig`
  - `plainsightai/openfilter-chatgpt-annotator` → `plainsightai/openfilter-chattag` (Docker Hub)
- Renamed frame metadata key `meta.chatgpt_annotator` → `meta.chattag`. The dict shape inside is unchanged; `labels.jsonl` is unchanged.
- Replaced env var `FILTER_CHATGPT_MODEL=gpt-4o-mini` with `FILTER_CHATTAG_MODEL=openai:gpt-4o-mini` (LangChain `provider:model` string).
- Removed env var `FILTER_CHATGPT_API_KEY` — set the provider's native env var (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_HOST`).
- Replaced `openai>=1.0.0` dependency with `langchain` + `langchain-openai` + `langchain-google-genai` + `langchain-anthropic` + `langchain-ollama` + `pydantic`.

### Unchanged
- PyPI distribution name remains `filter-chatgpt-annotator` (`pip install filter-chatgpt-annotator==0.3.0`).
- GitHub repository URL.

## v0.2.3 - 2026-04-23

### Changed
- Update the openfilter dependency to `>=0.1.30`, and align the CI workflow with the shared release gate (source-paths).
- Fix release workflow secret names: `PYPI_API_TOKEN` → `PLAINSIGHT_PYPI_TOKEN`, `DOCKERHUB_TOKEN` → `DOCKERHUB_ACCESS_TOKEN` (org-level secret names). Without this the PyPI / Docker Hub tokens resolved to empty and no package has been published since the migration.

## v0.2.2 - 2026-04-20

### Changed
- Remove redundant ci.yaml (shared workflow handles PR testing)
- Add push + pull_request triggers to create-release.yaml


## v0.2.1 - 2026-04-14

### Changed
- Add CI/CD workflows: create-release.yaml (Docker Hub publishing), ci.yaml (PR testing), security-scan.yaml (Grype)
- Update openfilter dependency to >=0.1.27
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
