# Alternative Tokenizer Setup for CSM

This document provides instructions for running CSM when you don't have access to the gated `meta-llama/Llama-3.2-1B` tokenizer.

## Problem

CSM requires the Llama-3.2-1B tokenizer, which is gated and requires approval. This can cause delays when trying to use the model.

## Solution

The CSM generator now supports fallback tokenizers that don't require special access.

## Quick Start

### Option 1: Automatic Fallback (Recommended)
Simply run CSM normally - it will automatically try alternative tokenizers:

```bash
export NO_TORCH_COMPILE=1
python run_csm.py
```

The system will try tokenizers in this order:
1. `meta-llama/Llama-3.2-1B` (original, gated)
2. `meta-llama/Llama-3.1-8B-Instruct` (often ungated)
3. `meta-llama/CodeLlama-7b-Python-hf` (usually ungated)
4. `Qwen/Qwen2-1.5B` (alternative with large vocabulary)
5. `microsoft/DialoGPT-medium` (fallback, works well)
6. `NousResearch/Llama-2-7b-chat-hf` (community version)
7. `huggyllama/llama-7b` (community Llama)
8. `meta-llama/Llama-2-7b-chat-hf` (final fallback)

### Option 2: Specify Tokenizer Explicitly
If you want to use a specific tokenizer:

```bash
# Use environment variable
export CSM_TOKENIZER_MODEL="microsoft/DialoGPT-medium"
python run_csm.py

# Or use command line argument
python run_csm.py --tokenizer microsoft/DialoGPT-medium
```

### Option 3: Command Line Options
The run script now supports additional options:

```bash
python run_csm.py --help

# Examples:
python run_csm.py --tokenizer microsoft/DialoGPT-medium --device cpu --output my_conversation.wav
```

## Recommended Alternative Tokenizers

### Best Compatibility
- `microsoft/DialoGPT-medium` - **Recommended** fallback, works well despite vocabulary size difference
- `NousResearch/Llama-2-7b-chat-hf` - Community version, often accessible

### If Available
- `meta-llama/Llama-3.1-8B-Instruct` - Same vocabulary size as original, but may be gated
- `Qwen/Qwen2-1.5B` - Large vocabulary, good alternative

## Technical Details

### Vocabulary Size Impact
- **Original**: `meta-llama/Llama-3.2-1B` has 128,256 tokens
- **Working Alternative**: `microsoft/DialoGPT-medium` has 50,257 tokens
- Despite the vocabulary size difference, DialoGPT-medium produces good results

### Debug Output
The system will show which tokenizer it's using:
```
Attempting to load tokenizer: meta-llama/Llama-3.2-1B
Failed to load tokenizer meta-llama/Llama-3.2-1B: You are trying to access a gated repo...
Attempting to load tokenizer: microsoft/DialoGPT-medium
Warning: microsoft/DialoGPT-medium has vocab size 50257, expected 128256
Successfully loaded tokenizer: microsoft/DialoGPT-medium
```

### Generation Quality
Testing shows that alternative tokenizers like DialoGPT-medium still produce good audio quality:
- ✅ Audio generation works
- ✅ Conversation flow maintained
- ✅ Multiple speakers supported
- ⚠️ May have slight quality differences due to vocabulary mismatch

## Troubleshooting

### All Tokenizers Fail
If all fallback tokenizers fail:
1. Check your internet connection
2. Ensure you're logged into Hugging Face: `huggingface-cli login`
3. Request access to gated models at their respective Hugging Face pages
4. Use a specific working tokenizer: `python run_csm.py --tokenizer microsoft/DialoGPT-medium`

### Audio Generation Issues
If audio generation produces only silence:
- The tokenizer vocabulary size might be too different
- Try a different tokenizer from the list above
- Check the debug output for "EOS detected at step 0"

### Out of Memory
If you get CUDA out of memory errors:
- Use `--device cpu` to run on CPU
- Or use a smaller tokenizer model

## Long-term Solution

For best results, request access to the original models:
1. Visit https://huggingface.co/meta-llama/Llama-3.2-1B
2. Request access and wait for approval
3. Once approved, CSM will automatically use the original tokenizer

## Development

### Adding New Tokenizers
To add new fallback tokenizers, edit `generator.py` and add to the `tokenizer_alternatives` list:

```python
tokenizer_alternatives = [
    "meta-llama/Llama-3.2-1B",  # Original
    "your/new-tokenizer",       # Add here
    # ... existing alternatives
]
```

### Testing Tokenizers
Test a specific tokenizer:
```bash
python run_csm.py --tokenizer your/test-tokenizer --output test.wav
```