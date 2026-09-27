.PHONY: all setup test data run validate package clean

PYTHON ?= .venv/bin/python

all: data run validate

setup:
	$(PYTHON) -m pip install -e code/business_entity_resolution

test:
	$(PYTHON) -m pytest code/business_entity_resolution/tests/ -v

data:
	$(PYTHON) -m ber.cli generate-synthetic --train-size 100 --test-size 50

run:
	$(PYTHON) -m ber.cli run --config configs/final.yaml

validate:
	$(PYTHON) utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

package:
	zip -r submission.zip output/matching_results.tsv output/candidate_pairs.tsv code/business_entity_resolution Documentation_template.md

clean:
	rm -rf output/*.tsv submission.zip
