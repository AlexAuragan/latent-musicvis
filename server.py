"""
Latent Space Explorer - FastAPI Server
Wraps Stable Audio VAE for encode/decode + UMAP projection

Environment Variables:
  VAE_CONFIG_PATH  - Path to VAE config JSON (default: stable_audio_2_0_vae.json)
  VAE_CKPT_PATH    - Path to VAE checkpoint (default: sao_vae_tune_100k_unwrapped.ckpt)
  PORT             - Server port (default: 8420)
"""

from contextlib import asynccontextmanager
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import torch
import torchaudio
import io
import json
import hashlib
from typing import Optional, Dict
import umap
import tempfile
from back import vae, device, VAE_CONFIG_PATH, VAE_CKPT_PATH, SAMPLE_RATE, load_vae, encode_audio_chunked, unload_vae, \
    MAX_CACHE_ENTRIES, SAMPLES_PER_LATENT, decode_audio_chunked

# Store original waveform for playback
current_waveform: Optional[torch.Tensor] = None

# Cache for encoded latents: hash -> (latents_np, projection, waveform, duration)
latent_cache: Dict[str, tuple] = {}

PORT = int(os.environ.get("PORT", "8420"))

@asynccontextmanager
async def startup(app: FastAPI):
    print(f"Server starting on port {PORT}")
    print(f"VAE config: {VAE_CONFIG_PATH}")
    print(f"VAE checkpoint: {VAE_CKPT_PATH}")
    print(f"Device: {device}")
    yield


app = FastAPI(title="Latent Space Explorer API", lifespan=startup)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayRequest(BaseModel):
    index: int


# Store current session's latents
current_latents: Optional[np.ndarray] = None
current_projection: Optional[np.ndarray] = None


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/explorer.html")


@app.post("/encode_stream")
async def encode_stream_endpoint(file: UploadFile = File(...)):
    """Upload audio, encode to latents, run UMAP, stream progress via SSE"""
    global current_latents, current_projection, current_waveform, vae, latent_cache

    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()

    async def generate():
        global current_latents, current_projection, current_waveform, vae, latent_cache

        try:
            # Check cache
            if file_hash in latent_cache:
                print(f"Cache hit for {file_hash[:8]}...")
                cached = latent_cache[file_hash]
                latents_np, projection_normalized, waveform, duration = cached

                current_latents = latents_np
                current_projection = projection_normalized
                current_waveform = waveform

                yield f"data: {json.dumps({'stage': 'cached'})}\n\n"
                yield f"data: {json.dumps({'stage': 'done', 'projection': projection_normalized.tolist(), 'latents': latents_np.tolist(), 'duration_seconds': duration, 'samples_per_latent': SAMPLES_PER_LATENT, 'num_latents': int(latents_np.shape[0])})}\n\n"
                return

            # Load audio
            audio_buffer = io.BytesIO(content)
            waveform, sr = torchaudio.load(audio_buffer)

            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)

            if waveform.shape[0] == 1:
                waveform = torch.cat([waveform, waveform], dim=0)
            elif waveform.shape[0] > 2:
                waveform = waveform[:2]

            waveform = waveform / (waveform.abs().max() + 1e-6)
            current_waveform = waveform

            yield f"data: {json.dumps({'stage': 'loading_vae'})}\n\n"

            if vae is None:
                load_vae()

            yield f"data: {json.dumps({'stage': 'encoding'})}\n\n"

            latents = encode_audio_chunked(waveform, chunk_seconds=10.0)
            latents_np = latents[0].cpu().numpy().T
            current_latents = latents_np

            print(
                f"Encoded {waveform.shape[1]} samples -> {latents_np.shape[0]} latents"
            )

            yield f"data: {json.dumps({'stage': 'unloading_vae'})}\n\n"
            unload_vae()

            yield f"data: {json.dumps({'stage': 'umap', 'num_latents': int(latents_np.shape[0])})}\n\n"

            # UMAP
            if latents_np.shape[0] < 5:
                projection = latents_np[:, :3]
            else:
                n_pts = latents_np.shape[0]
                reducer = umap.UMAP(
                    n_components=3,
                    n_neighbors=min(30, n_pts - 1),
                    min_dist=0.05,
                    n_epochs=1000,
                    metric="euclidean",
                    spread=1.0,
                    random_state=42,
                )
                projection = reducer.fit_transform(latents_np)

            # Center and scale
            center = projection.mean(axis=0)
            projection_centered = projection - center
            max_abs = np.abs(projection_centered).max() + 1e-6
            projection_normalized = projection_centered / max_abs

            current_projection = projection_normalized

            # Cache
            duration = float(waveform.shape[1] / SAMPLE_RATE)
            if len(latent_cache) >= MAX_CACHE_ENTRIES:
                oldest_key = next(iter(latent_cache))
                del latent_cache[oldest_key]
            latent_cache[file_hash] = (
                latents_np,
                projection_normalized,
                waveform,
                duration,
            )

            yield f"data: {json.dumps({'stage': 'done', 'projection': projection_normalized.tolist(), 'latents': latents_np.tolist(), 'duration_seconds': duration, 'samples_per_latent': SAMPLES_PER_LATENT, 'num_latents': int(latents_np.shape[0])})}\n\n"

        except Exception as e:
            import traceback

            traceback.print_exc()
            yield f"data: {json.dumps({'stage': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/audio_full")
async def audio_full_endpoint():
    """Return the full loaded audio as WAV"""
    global current_waveform

    if current_waveform is None:
        raise HTTPException(status_code=400, detail="No audio loaded")

    waveform = current_waveform.cpu()

    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        # TorchCodec infers format from the .wav extension
        torchaudio.save(tmp.name, waveform, SAMPLE_RATE)
        tmp.seek(0)
        data = tmp.read()

    return Response(content=data, media_type="audio/wav")


@app.post("/play")
async def play_endpoint(request: PlayRequest):
    """Play original audio chunk for a latent index (2048 samples)"""
    global current_waveform

    if current_waveform is None:
        raise HTTPException(status_code=400, detail="No audio loaded")

    idx = request.index
    start = idx * SAMPLES_PER_LATENT
    end = start + SAMPLES_PER_LATENT

    if end > current_waveform.shape[1]:
        raise HTTPException(status_code=400, detail=f"Index {idx} out of range")

    chunk = current_waveform[:, start:end]
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        torchaudio.save(tmp.name, chunk, SAMPLE_RATE)
        tmp.seek(0)
        data = tmp.read()

    return Response(content=data, media_type="audio/wav")


@app.post("/resynth")
async def resynth(file: UploadFile = File(...)):
    """
    Latent Resynthesis: use current audio's latents as codebook,
    encode uploaded file, replace each latent with nearest neighbor,
    decode and return resynthesized audio.
    """
    global current_latents, vae

    if current_latents is None:
        raise HTTPException(
            status_code=400, detail="No codebook loaded - upload a source audio first"
        )

    try:
        audio_bytes = await file.read()
        buffer = io.BytesIO(audio_bytes)
        waveform, sr = torchaudio.load(buffer)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]

        print(f"Resynth: encoding target audio {waveform.shape}")

        if vae is None:
            load_vae()

        target_latents = encode_audio_chunked(waveform, chunk_seconds=10.0)
        target_np = target_latents[0].cpu().numpy().T

        print(
            f"Resynth: {target_np.shape[0]} target latents, {current_latents.shape[0]} codebook latents"
        )

        codebook = current_latents

        # Nearest neighbor lookup
        resynth_indices = []
        for i in range(target_np.shape[0]):
            dists = np.sum((codebook - target_np[i : i + 1]) ** 2, axis=1)
            nearest_idx = np.argmin(dists)
            resynth_indices.append(nearest_idx)

        resynth_indices = np.array(resynth_indices)
        print(f"Resynth: mapped to indices (unique: {len(np.unique(resynth_indices))})")

        resynth_latents = codebook[resynth_indices]
        resynth_tensor = torch.from_numpy(resynth_latents.T).unsqueeze(0).float()

        print("Resynth: decoding in chunks...")
        resynth_audio = decode_audio_chunked(resynth_tensor, chunk_latents=200)

        unload_vae()

        audio_out = resynth_audio[0].cpu()

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            torchaudio.save(tmp.name, audio_out, SAMPLE_RATE)
            tmp.seek(0)
            data = tmp.read()

        print(f"Resynth: complete, output {audio_out.shape[1]} samples")

        return Response(content=data, media_type="audio/wav")

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vae_loaded": vae is not None,
        "device": device,
        "latents_loaded": current_latents is not None,
        "num_latents": current_latents.shape[0] if current_latents is not None else 0,
    }


# Serve static files (explorer.html)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
