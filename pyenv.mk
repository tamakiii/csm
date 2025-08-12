.PHONY: help setup teardown activate install

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
	.venv/bin/pip install --upgrade pip setuptools wheel

install: .venv
	@echo "Installing dependencies..."
	-.venv/bin/pip install torch==2.6.0 torchaudio==2.6.0
	-.venv/bin/pip install tokenizers==0.21.0 transformers==4.49.0 huggingface_hub==0.28.1
	-.venv/bin/pip install moshi==0.2.2
	-.venv/bin/pip install torchao==0.9.0
	-.venv/bin/pip install git+https://github.com/SesameAILabs/silentcipher@master
	@echo "Skipping torchtune due to sentencepiece dependency issue with Python 3.13"
	@echo "Installation complete (some packages may have been skipped)"
