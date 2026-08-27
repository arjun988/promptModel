.PHONY: install install-demo doctor test lint clean

install:
	pip install -U pip
	pip install -e .

install-demo:
	pip install -e ".[demo,dev]"

doctor:
	promptforge doctor

init:
	promptforge init

test:
	pytest -q

lint:
	ruff check src scripts tests

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
