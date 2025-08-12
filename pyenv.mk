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
	python -m venv .venv
