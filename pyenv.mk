.PHONY: help setup teardown activate

help:
	@(cat $(firstword $(MAKEIFLE_LIST)))

setup: \
	.venv

teardown:
	rm -rf .venv

activate: .venv
	@bash --init-file <(echo "source .venv/bin/activate")

.venv:
	pyenv exec python -m venv .venv
	.venv/bin/pip install --upgrade pip setuptools wheel
