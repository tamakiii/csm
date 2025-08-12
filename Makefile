.PHONY: help install

help:
	@(cat $(firstword $(MAKEIFLE_LIST)))

install:
	@echo "Installing dependencies for Python 3.11..."
	pip install torch==2.6.0 torchaudio==2.6.0
	pip install tokenizers==0.21.0 transformers==4.49.0 huggingface_hub==0.28.1
	pip install torchao==0.9.0
	pip install git+https://github.com/SesameAILabs/silentcipher@master
	pip install moshi
	pip install torchtune==0.5.0

