.PHONY: all setup test data train predict validate package clean

PYTHON ?= python3

all: test validate

setup:
	$(PYTHON) -m pip install -e .

test:
	$(PYTHON) -m pytest tests/ -v

data:
	$(PYTHON) -m ber.cli generate-synthetic --train-size 100 --test-size 50

train:
	$(PYTHON) -m ber.cli run --config configs/final.yaml

validate:
	$(PYTHON) ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test

package:
	zip -r ../../submission.zip ../../output/matching_results.tsv ../../output/candidate_pairs.tsv . ../../Documentation_template.md

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache experiments/__pycache__
