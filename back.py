"""
Latent Space Explorer - FastAPI Server
Wraps Stable Audio VAE for encode/decode + UMAP projection

Environment Variables:
  VAE_CONFIG_PATH  - Path to VAE config JSON (default: stable_audio_2_0_vae.json)
  VAE_CKPT_PATH    - Path to VAE checkpoint (default: sao_vae_tune_100k_unwrapped.ckpt)
"""

import os
import numpy as np
import torch
import json
import gc
from typing import Optional, Dict


# ============ CONFIG ============
VAE_CONFIG_PATH = os.environ.get("VAE_CONFIG_PATH", "stable_audio_2_0_vae.json")
VAE_CKPT_PATH = os.environ.get("VAE_CKPT_PATH", "sao_vae_tune_100k_unwrapped.ckpt")

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 2048
LATENT_DIM = 64

# Global VAE model
vae = None
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CACHE_ENTRIES = 10


def find_best_offset(waveform: torch.Tensor, max_offset: int = SAMPLES_PER_LATENT) -> int:
    """
    waveform: [2, samples]
    returns best integer offset in [0, max_offset)
    """
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # [1, 2, samples]

    batch, channels, total_samples = waveform.shape
    best_score = float("inf")
    best_offset = 0
    edge = 32  # how many samples from each side of the boundary to compare

    for offset in range(max_offset):
        usable = total_samples - offset
        if usable <= SAMPLES_PER_LATENT * 2:
            break  # not enough data to evaluate

        num_frames = usable // SAMPLES_PER_LATENT
        end = offset + num_frames * SAMPLES_PER_LATENT

        # [1, 2, num_frames, frame_len]
        frames = waveform[:, :, offset:end].view(batch, channels, num_frames, SAMPLES_PER_LATENT)

        # Compare last edge samples of frame n with first edge samples of frame n+1
        left = frames[:, :, :-1, -edge:]
        right = frames[:, :, 1:, :edge]

        diff = left - right
        score = diff.pow(2).mean().item()  # lower = smoother boundaries

        if score < best_score:
            best_score = score
            best_offset = offset

    return best_offset

def load_vae():
    """Load the VAE model from configured paths"""
    global vae

    try:
        from stable_audio_tools.models.factory import create_model_from_config
        from stable_audio_tools.models.utils import (
            copy_state_dict,
            load_ckpt_state_dict,
        )

        if not os.path.exists(VAE_CONFIG_PATH):
            raise FileNotFoundError(f"VAE config not found: {VAE_CONFIG_PATH}")
        if not os.path.exists(VAE_CKPT_PATH):
            raise FileNotFoundError(f"VAE checkpoint not found: {VAE_CKPT_PATH}")

        model_config = json.load(open(VAE_CONFIG_PATH))
        vae = create_model_from_config(model_config)
        copy_state_dict(vae, load_ckpt_state_dict(VAE_CKPT_PATH))
        vae.to(device).eval().requires_grad_(False)
        print(f"VAE loaded on {device}")

    except ImportError:
        print("WARNING: stable_audio_tools not installed. Using mock encoder/decoder.")
        vae = None
    except Exception as e:
        print(f"WARNING: Failed to load VAE: {e}. Using mock encoder/decoder.")
        vae = None


def unload_vae():
    """Unload VAE to free VRAM"""
    global vae
    if vae is not None:
        del vae
        vae = None
        torch.cuda.empty_cache()
        gc.collect()
        print("VAE unloaded, VRAM freed")


def encode_audio_chunked(
    waveform: torch.Tensor, chunk_seconds: float = 10.0
) -> torch.Tensor:
    """
    Encode audio waveform to latents in chunks to reduce VRAM usage
    Input: [2, samples]
    Output: [1, 64, num_latents]
    """
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    if vae is None:
        return encode_audio_mock(waveform)

    total_samples = waveform.shape[2]
    chunk_samples = int(chunk_seconds * SAMPLE_RATE)
    chunk_samples = (chunk_samples // SAMPLES_PER_LATENT) * SAMPLES_PER_LATENT

    all_latents = []

    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunk = waveform[:, :, start:end]

        if chunk.shape[2] < SAMPLES_PER_LATENT:
            pad = SAMPLES_PER_LATENT - chunk.shape[2]
            chunk = torch.nn.functional.pad(chunk, (0, pad))
        elif chunk.shape[2] % SAMPLES_PER_LATENT != 0:
            pad = SAMPLES_PER_LATENT - (chunk.shape[2] % SAMPLES_PER_LATENT)
            chunk = torch.nn.functional.pad(chunk, (0, pad))

        chunk_gpu = chunk.to(device)
        with torch.no_grad():
            latents = vae.encode(chunk_gpu)
        all_latents.append(latents.cpu())

        del chunk_gpu, latents
        torch.cuda.empty_cache()

    return torch.cat(all_latents, dim=2).to(device)


def encode_audio_mock(waveform: torch.Tensor) -> torch.Tensor:
    """Mock encoding when VAE not available - extracts audio features as latents"""
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.to(device)
    batch, channels, samples = waveform.shape
    num_latents = samples // SAMPLES_PER_LATENT

    latents = torch.zeros(batch, LATENT_DIM, num_latents, device=device)
    for i in range(num_latents):
        chunk = waveform[:, :, i * SAMPLES_PER_LATENT : (i + 1) * SAMPLES_PER_LATENT]
        latents[:, 0, i] = chunk.mean()
        latents[:, 1, i] = chunk.std()
        latents[:, 2, i] = chunk.abs().max()
        fft = torch.fft.rfft(chunk.mean(dim=1), dim=-1)
        mag = fft.abs()
        for j in range(min(30, LATENT_DIM - 3)):
            bin_start = j * len(mag[0]) // 30
            bin_end = (j + 1) * len(mag[0]) // 30
            latents[:, j + 3, i] = mag[:, bin_start:bin_end].mean()
        latents[:, 33:, i] = torch.randn(
            batch, LATENT_DIM - 33, device=device
        ) * chunk.std().unsqueeze(-1)

    return latents


def decode_latents(latents: torch.Tensor) -> torch.Tensor:
    """Decode latents to audio. Input: [batch, 64, num_latents], Output: [batch, 2, samples]"""
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)

    latents = latents.to(device)

    if vae is not None:
        with torch.no_grad():
            audio = vae.decode(latents)
        return audio
    else:
        return decode_latents_mock(latents)


def decode_latents_mock(latents: torch.Tensor) -> torch.Tensor:
    """Mock decoding when VAE not available"""
    latents = latents.to(device)

    batch, dim, num_latents = latents.shape
    samples = num_latents * SAMPLES_PER_LATENT

    audio = torch.zeros(batch, 2, samples, device=device)
    t = torch.linspace(
        0, num_latents * SAMPLES_PER_LATENT / SAMPLE_RATE, samples, device=device
    )

    for i in range(num_latents):
        start = i * SAMPLES_PER_LATENT
        end = (i + 1) * SAMPLES_PER_LATENT
        t_chunk = t[start:end]

        chunk = torch.zeros(batch, 2, SAMPLES_PER_LATENT, device=device)
        base_freq = 110 + latents[:, 0, i : i + 1].cpu().numpy()[0, 0] * 220

        for h in range(8):
            freq = base_freq * (h + 1)
            amp = 0.3 / (h + 1) * (1 + latents[:, min(h + 3, 63), i : i + 1])
            phase = latents[:, min(h + 10, 63), i : i + 1] * np.pi
            wave = amp * torch.sin(2 * np.pi * freq * t_chunk + phase)
            chunk[:, 0, :] += wave.squeeze()
            chunk[:, 1, :] += (
                wave.squeeze()
                * (
                    0.8 + 0.4 * torch.tanh(latents[:, min(h + 20, 63), i : i + 1])
                ).squeeze()
            )

        chunk = chunk / (chunk.abs().max() + 1e-6) * 0.7
        audio[:, :, start:end] = chunk

    return audio


def decode_audio_chunked(
    latents: torch.Tensor, chunk_latents: int = 200
) -> torch.Tensor:
    """Decode latents in chunks to avoid VRAM explosion. ~200 latents = ~10 seconds"""
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)

    if vae is None:
        return decode_latents_mock(latents)

    num_latents = latents.shape[2]
    chunks = []

    for start in range(0, num_latents, chunk_latents):
        end = min(start + chunk_latents, num_latents)
        chunk = latents[:, :, start:end].to(device)

        with torch.no_grad():
            audio_chunk = vae.decode(chunk)

        chunks.append(audio_chunk.cpu())
        torch.cuda.empty_cache()

    return torch.cat(chunks, dim=2)
