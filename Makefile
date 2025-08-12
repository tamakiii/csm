.PHONY: help install

help:
	@(cat $(firstword $(MAKEIFLE_LIST)))

install:
	@echo "Installing dependencies (skipping packages incompatible with Python 3.13)..."
	-pip install torch==2.6.0 torchaudio==2.6.0
	-pip install tokenizers==0.21.0 transformers==4.49.0 huggingface_hub==0.28.1
	-pip install torchao==0.9.0
	-pip install git+https://github.com/SesameAILabs/silentcipher@master
	@echo "Skipping moshi and torchtune due to sentencepiece dependency issue with Python 3.13"

