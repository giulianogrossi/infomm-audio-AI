"""Lesson 5: pretrained generation, with training-free offline analysis.

Runs the AudioLDM-S text-to-audio pipeline on three reproducible trials and
writes the generated clips with their provenance. The offline reader opens the
precomputed materials in ``lab-2026/materiali/05-generazione``.
"""

import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import MATERIALS_DIR

MODEL_ID = "cvssp/audioldm-s-full-v2"
REVISION = "feeb3d14203495a4b6ac0893cbdedb2159b4819c"

#: The three generation trials: same prompt at two seeds, then a different room.
PROMPTS = [
    dict(nome="A", seed=42, prompt="A single short dog bark in a quiet dry room."),
    dict(nome="B", seed=43, prompt="A single short dog bark in a quiet dry room."),
    dict(nome="C", seed=42, prompt="A single short dog bark in a quiet reverberant room."),
]

#: Directory of the precomputed clips.
OFFLINE_DIR = MATERIALS_DIR / "05-generazione"


def generate(folder, device="cpu", steps=25):
    """Generate the three clips with AudioLDM-S and write them to ``folder``.

    A new folder is required, so previous results are never overwritten. Each
    clip comes with a JSON file recording model revision, parameters and
    package versions.

    Parameters
    ----------
    folder : str or pathlib.Path
        Destination folder (must not exist).
    device : str, default "cpu"
        "cpu", "cuda" or "mps".
    steps : int, default 25
        Number of inference steps.

    Returns
    -------
    pathlib.Path
        The folder written.

    Examples
    --------
    >>> al.generate("tmp/generazione-01", device="cpu", steps=25)
    """
    import torch
    from diffusers import AudioLDMPipeline
    from huggingface_hub import snapshot_download

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    snapshot = snapshot_download(MODEL_ID, revision=REVISION,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors"])
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = AudioLDMPipeline.from_pretrained(snapshot, torch_dtype=dtype,
         use_safetensors=True, local_files_only=True).to(device)
    sr = int(pipe.vocoder.config.sampling_rate)
    packages = {p: version(p) for p in ["torch", "diffusers", "transformers", "numpy",
                                        "librosa", "soundfile", "huggingface-hub"]}
    for trial in PROMPTS:
        start = time.perf_counter()
        with torch.inference_mode():
            out = pipe(prompt=trial["prompt"], audio_length_in_s=5.0,
                num_inference_steps=steps, guidance_scale=2.5,
                generator=torch.Generator(device="cpu").manual_seed(trial["seed"]))
        if device == "mps":
            torch.mps.synchronize()
        if device == "cuda":
            torch.cuda.synchronize()
        y = np.asarray(out.audios[0], dtype=np.float32)
        if y.ndim != 1 or not y.size or not np.isfinite(y).all():
            raise ValueError("Waveform generata non valida.")
        name = trial["nome"]
        sf.write(folder / f"{name}.wav", y, sr, subtype="FLOAT")
        meta = {**trial, "model_id": MODEL_ID, "revision": REVISION,
                "sampling_rate": sr, "duration_requested_s": 5.0,
                "duration_actual_s": len(y) / sr,
                "steps": steps, "guidance_scale": 2.5, "negative_prompt": None,
                "device": device, "dtype": str(dtype), "packages": packages,
                "scheduler": type(pipe.scheduler).__name__,
                "scheduler_config": dict(pipe.scheduler.config),
                "hardware": platform.machine(), "python": platform.python_version(),
                "inference_seconds": time.perf_counter() - start,
                "peak_raw": float(np.max(np.abs(y)))}
        (folder / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        print(name, meta["inference_seconds"], "s", flush=True)
    return folder


def load_samples(folder=OFFLINE_DIR):
    """Read the precomputed clips with their provenance.

    Parameters
    ----------
    folder : str or pathlib.Path, default OFFLINE_DIR
        The folder written by ``generate()``.

    Returns
    -------
    list of (str, numpy.ndarray, int, dict)
        For each clip: name, signal, sampling rate and metadata.

    Examples
    --------
    >>> for name, y, sr, meta in al.load_samples():
    ...     print(name, round(len(y) / sr, 2), "s")
    """
    clips = []
    for name in ["A", "B", "C"]:
        p = Path(folder)
        if not (p / f"{name}.wav").is_file() or not (p / f"{name}.json").is_file():
            raise FileNotFoundError(
                f"Materiali incompleti in {p}: servono A/B/C.wav e JSON autentici.")
        y, sr = sf.read(p / f"{name}.wav", dtype="float32")
        meta = json.loads((p / f"{name}.json").read_text())
        if y.ndim != 1 or not y.size or not np.isfinite(y).all() or sr != meta["sampling_rate"]:
            raise ValueError(f"Audio o metadati non validi: {name}")
        clips.append((name, y, sr, meta))
    return clips


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int, default=25)
    args = parser.parse_args()
    generate(args.output, args.device, args.steps)
