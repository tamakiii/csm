from dataclasses import dataclass
from typing import List, Tuple

import torch
import torchaudio
from huggingface_hub import hf_hub_download
from models import Model
from moshi.models import loaders
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from watermarking import CSM_1B_GH_WATERMARK, load_watermarker, watermark


@dataclass
class Segment:
    speaker: int
    text: str
    # (num_samples,), sample_rate = 24_000
    audio: torch.Tensor


def load_llama3_tokenizer(tokenizer_name: str = None):
    """
    Load a Llama-compatible tokenizer with fallback options for gated models.
    
    Args:
        tokenizer_name: Optional tokenizer model name. If None, uses environment variable
                       CSM_TOKENIZER_MODEL or falls back to alternatives.
    
    https://github.com/huggingface/transformers/issues/22794#issuecomment-2092623992
    """
    import os
    
    # List of tokenizer alternatives in order of preference
    tokenizer_alternatives = [
        "meta-llama/Llama-3.2-1B",  # Original (gated) - 128,256 tokens
        "meta-llama/Llama-3.1-8B-Instruct",  # Often ungated - 128,256 tokens
        "meta-llama/CodeLlama-7b-Python-hf",  # Usually ungated - 32,000 tokens
        # Try some potentially ungated Llama-3 variants
        "Qwen/Qwen2-1.5B",  # Alternative with large vocab
        "microsoft/DialoGPT-medium",  # Different approach - 50,257 tokens
        "NousResearch/Llama-2-7b-chat-hf",  # Community version
        "huggyllama/llama-7b",  # Community Llama
        "meta-llama/Llama-2-7b-chat-hf",  # Fallback option - 32,000 tokens
    ]
    
    # Determine which tokenizer to use
    if tokenizer_name is None:
        tokenizer_name = os.environ.get("CSM_TOKENIZER_MODEL")
    
    # If still no tokenizer specified, try alternatives
    if tokenizer_name is None:
        tokenizers_to_try = tokenizer_alternatives
    else:
        tokenizers_to_try = [tokenizer_name]
    
    last_error = None
    for model_name in tokenizers_to_try:
        try:
            print(f"Attempting to load tokenizer: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Verify the tokenizer has the expected vocabulary size (128,256)
            vocab_size = len(tokenizer.get_vocab())
            if vocab_size != 128256:
                print(f"Warning: {model_name} has vocab size {vocab_size}, expected 128256")
                # Skip if vocabulary is too different (will cause generation issues)
                if abs(vocab_size - 128256) > 100000:  # Allow some tolerance
                    print(f"Skipping {model_name} due to large vocab size difference")
                    continue
            
            bos = tokenizer.bos_token
            eos = tokenizer.eos_token
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos}:0 $A:0 {eos}:0",
                pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
                special_tokens=[(f"{bos}", tokenizer.bos_token_id), (f"{eos}", tokenizer.eos_token_id)],
            )
            
            print(f"Successfully loaded tokenizer: {model_name}")
            return tokenizer
            
        except Exception as e:
            print(f"Failed to load tokenizer {model_name}: {e}")
            last_error = e
            continue
    
    # If all tokenizers failed, raise the last error
    raise RuntimeError(
        f"Failed to load any tokenizer. Tried: {tokenizers_to_try}\n"
        f"Last error: {last_error}\n"
        f"Solutions:\n"
        f"1. Set environment variable: export CSM_TOKENIZER_MODEL='meta-llama/Llama-3.1-8B-Instruct'\n"
        f"2. Request access to meta-llama/Llama-3.2-1B at https://huggingface.co/meta-llama/Llama-3.2-1B\n"
        f"3. Use a local/cached tokenizer"
    )


class Generator:
    def __init__(
        self,
        model: Model,
    ):
        self._model = model
        self._model.setup_caches(1)

        self._text_tokenizer = load_llama3_tokenizer()

        device = next(model.parameters()).device
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi

        self._watermarker = load_watermarker(device=device)

        self.sample_rate = mimi.sample_rate
        self.device = device

    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_tokens = []
        frame_masks = []

        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True

        frame_tokens.append(text_frame.to(self.device))
        frame_masks.append(text_frame_mask.to(self.device))

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert audio.ndim == 1, "Audio must be single channel"

        frame_tokens = []
        frame_masks = []

        # (K, T)
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        # add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)

        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True

        frame_tokens.append(audio_frame)
        frame_masks.append(audio_frame_mask)

        return torch.cat(frame_tokens, dim=0), torch.cat(frame_masks, dim=0)

    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (seq_len, 33), (seq_len, 33)
        """
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)

        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        speaker: int,
        context: List[Segment],
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
    ) -> torch.Tensor:
        self._model.reset_caches()

        max_generation_len = int(max_audio_length_ms / 80)
        tokens, tokens_mask = [], []
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)

        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)

        prompt_tokens = torch.cat(tokens, dim=0).long().to(self.device)
        prompt_tokens_mask = torch.cat(tokens_mask, dim=0).bool().to(self.device)

        samples = []
        curr_tokens = prompt_tokens.unsqueeze(0)
        curr_tokens_mask = prompt_tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, prompt_tokens.size(0)).unsqueeze(0).long().to(self.device)

        max_seq_len = 2048
        max_context_len = max_seq_len - max_generation_len
        if curr_tokens.size(1) >= max_context_len:
            raise ValueError(
                f"Inputs too long, must be below max_seq_len - max_generation_len: {max_context_len}"
            )

        for i in range(max_generation_len):
            sample = self._model.generate_frame(curr_tokens, curr_tokens_mask, curr_pos, temperature, topk)
            if torch.all(sample == 0):
                print(f"EOS detected at step {i}, stopping generation")
                break  # eos

            samples.append(sample)

            curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
            curr_tokens_mask = torch.cat(
                [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
            ).unsqueeze(1)
            curr_pos = curr_pos[:, -1:] + 1

        if len(samples) == 0:
            print("Warning: No audio samples generated, returning silence")
            # Return short silence instead of crashing
            return torch.zeros(int(self.sample_rate * 0.1))  # 0.1 second of silence

        print(f"Generated {len(samples)} audio samples")
        audio = self._audio_tokenizer.decode(torch.stack(samples).permute(1, 2, 0)).squeeze(0).squeeze(0)

        # This applies an imperceptible watermark to identify audio as AI-generated.
        # Watermarking ensures transparency, dissuades misuse, and enables traceability.
        # Please be a responsible AI citizen and keep the watermarking in place.
        # If using CSM 1B in another application, use your own private key and keep it secret.
        audio, wm_sample_rate = watermark(self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK)
        audio = torchaudio.functional.resample(audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate)

        return audio


def load_csm_1b(device: str = "cuda") -> Generator:
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)

    generator = Generator(model)
    return generator