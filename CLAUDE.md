# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSM (Conversational Speech Model) is a speech generation model from Sesame that generates RVQ audio codes from text and audio inputs. It uses a Llama backbone architecture with a smaller audio decoder that produces Mimi audio codes.

## Development Environment Setup

### Python Version Requirements
- **Use Python 3.11** - The project requires Python 3.11 due to sentencepiece compatibility issues with Python 3.13
- The `.python-version` file controls the Python version via pyenv

### Virtual Environment Management
The project uses a custom `pyenv.mk` Makefile for virtual environment management:

```bash
# Create and setup virtual environment
make -f pyenv.mk setup

# Activate the virtual environment
make -f pyenv.mk activate

# Remove virtual environment
make -f pyenv.mk teardown
```

### Installing Dependencies
After activating the virtual environment:

```bash
# Install all dependencies from requirements.txt
make install
```

Note: The main `Makefile` has been modified to handle Python 3.13 compatibility issues by skipping incompatible packages (moshi and torchtune), but with Python 3.11 all packages should install correctly.

## Core Architecture

### Key Components

1. **`generator.py`** - Main generation logic
   - `Generator` class: Handles model setup, tokenization, and audio generation
   - `Segment` dataclass: Represents audio context segments with speaker, text, and audio
   - Integrates text tokenizer (Llama3), audio tokenizer (Mimi), and watermarking

2. **`models.py`** - Model architecture definitions
   - Defines CSM model variants (1B and 100M parameters)
   - Uses torchtune's Llama 3.2 implementation
   - Custom `Model` class wraps the transformer with audio-specific modifications

3. **`run_csm.py`** - Example usage script
   - Demonstrates conversational audio generation between two speakers
   - Loads pre-defined speaker prompts from Hugging Face
   - Shows how to use context segments for better generation quality

4. **`watermarking.py`** - Audio watermarking functionality
   - Integrates with SilentCipher for watermarking generated audio

### Model Requirements
The system requires access to:
- `meta-llama/Llama-3.2-1B` - Base language model
- `sesame/csm-1b` - CSM-specific weights and prompts
- `kyutai/mimi` - Audio tokenizer model

## Running the Project

### Basic Usage
```bash
# Disable lazy compilation (required)
export NO_TORCH_COMPILE=1

# Login to Hugging Face (required for model access)
huggingface-cli login

# Run example conversation generation
python run_csm.py
```

### Key Configuration
- The model supports CUDA, MPS (Apple Silicon), and CPU devices
- Audio sample rate is 24,000 Hz
- Maximum sequence length is 2048 tokens

## Important Dependencies

- **torch/torchaudio**: Core deep learning framework
- **transformers**: Hugging Face transformers library
- **moshi**: Audio codec models (Mimi)
- **torchtune**: Meta's training utilities for Llama models
- **silentcipher**: Audio watermarking library

## Development Notes

### Known Issues
- sentencepiece has compatibility issues with Python 3.13+
- Windows requires `triton-windows` instead of `triton`
- Requires `ffmpeg` for some audio operations

### Testing Audio Generation
When testing generation, always provide context segments for best results. The model performs poorly without context and will use random speaker identities.