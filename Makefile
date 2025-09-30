# ---------------------------------
# Repo-specific variables
# ---------------------------------

IMAGE ?= us-west1-docker.pkg.dev/plainsightai-prod/oci/filter-chatgpt-annotator
MODEL_IMAGE ?= us-west1-docker.pkg.dev/plainsightai-prod/oci/filter-chatgpt-annotator-model

# Define these variables for consistency in the repo
REPO_NAME ?= filter-chatgpt-annotator
REPO_NAME_SNAKECASE ?= filter_chatgpt_annotator
REPO_NAME_PASCALCASE ?= FilterChatgptAnnotator

# Unique pipeline configuration for this repo
PIPELINE := \
	- VideoIn \
		--sources file://data/sample-video.mp4!loop \
		--outputs 'tcp://*:5550' \
  - filter_chatgpt_annotator.filter.FilterChatgptAnnotator \
      --sources 'tcp://localhost:5550' \
      --outputs 'tcp://*:5552' \
      --chatgpt_api_key 'sk-open-ai-key' \
      --prompt './prompts/simple_salad_prompt_bb_v3.txt' \
      --output_schema '{"avocado":{"present":false,"confidence":0.0,"bbox":null},"lettuce":{"present":false,"confidence":0.0,"bbox":null},"tomato":{"present":false,"confidence":0.0,"bbox":null}}' \
      --mq_log pretty \
  - Webvis \
      --sources 'tcp://localhost:5552'

# ---------------------------------
# Repo-specific targets
# ---------------------------------

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install package with dev dependencies
	pip install -e .[dev] \
		--index-url https://python.openfilter.io/simple \
		--extra-index-url https://pypi.org/simple

.PHONY: run
run:  ## Run locally with supporting Filters in other processes
	openfilter run ${PIPELINE}

.PHONY: test
test:  ## Run unit tests
	pytest -vv -s tests/ --junitxml=results/pytest-results.xml

.PHONY: test-coverage
test-coverage:  ## Run unit tests and generate coverage report
	@mkdir -p Reports
	@pytest -vv --cov=tests --junitxml=Reports/coverage.xml --cov-report=json:Reports/coverage.json -s tests/
	@jq -r '["File Name", "Statements", "Missing", "Coverage%"], (.files | to_entries[] | [.key, .value.summary.num_statements, .value.summary.missing_lines, .value.summary.percent_covered_display]) | @csv'  Reports/coverage.json >  Reports/coverage_report.csv
	@jq -r '["TOTAL", (.totals.num_statements // 0), (.totals.missing_lines // 0), (.totals.percent_covered_display // "0")] | @csv'  Reports/coverage.json >>  Reports/coverage_report.csv

.PHONY: build-wheel
build-wheel:  ## Build python wheel
	python -m pip install setuptools build wheel twine setuptools-scm --index-url https://pypi.org/simple
	python -m build --wheel

.PHONY: clean
clean:  ## Delete all generated files and directories
	sudo rm -rf build/ cache/ dist/ $(REPO_NAME_SNAKECASE).egg-info/ telemetry/
	find . -name __pycache__ -type d -exec rm -rf {} +