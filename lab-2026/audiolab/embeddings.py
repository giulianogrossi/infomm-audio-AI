"""CLAP embeddings for the offline demo of chapter 5.

Runs the pretrained CLAP model on three course sounds and four text phrases,
and saves the audio and text embeddings used by the lesson.
"""

import csv
import inspect
import json
from pathlib import Path

from .paths import AUDIO_DIR, DATASETS_DIR, MATERIALS_DIR, PROJECT_ROOT

CLAP_MODEL_ID = "laion/clap-htsat-unfused"

#: Directory of the precomputed embeddings.
OFFLINE_DIR = MATERIALS_DIR / "05-clap"

#: The text phrases compared with the sounds.
PHRASES = [
    "The sound of a person speaking.",
    "The sound of a guitar playing a note.",
    "The sound of a dog barking.",
    "The sound of a siren.",
]


def prepare_clap_embeddings(output_dir=None):
    """Compute and save the CLAP embeddings of the demo sounds and phrases.

    Writes ``embedding.npz`` (keys "audio" and "testo") and
    ``provenienza.json`` in the output folder.

    Parameters
    ----------
    output_dir : str or pathlib.Path or None, default None
        Destination folder (default: ``lab-2026/materiali/05-clap``).

    Returns
    -------
    pathlib.Path
        The folder written.

    Examples
    --------
    >>> al.prepare_clap_embeddings()
    """
    import numpy as np
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor
    from huggingface_hub import model_info

    out_dir = Path(output_dir or OFFLINE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = CLAP_MODEL_ID
    revision = model_info(model_id).sha
    processor = ClapProcessor.from_pretrained(model_id, revision=revision)
    model = ClapModel.from_pretrained(model_id, revision=revision, weights_only=True).eval()
    torch.set_num_threads(4)
    rows = list(csv.DictReader(
        (DATASETS_DIR / "UrbanSound8K/metadata/UrbanSound8K.csv").open()))
    bark = next(x["filename"] for x in rows if x["classID"] == "3")
    files = [AUDIO_DIR / "speech1.wav", AUDIO_DIR / "Guitar_C4.wav", DATASETS_DIR / bark]
    sr = processor.feature_extractor.sampling_rate
    clips = [librosa.load(p, sr=sr, mono=True, duration=5)[0] for p in files]
    # transformers 4.56 exposes audios; later versions may use audio.
    keyword = "audios" if "audios" in inspect.signature(processor.__call__).parameters else "audio"
    inputs = processor(**{keyword: clips}, text=PHRASES, sampling_rate=sr, padding=True,
                        return_tensors="pt")
    with torch.inference_mode():
        out = model(**inputs)
    audio = out.audio_embeds.numpy()
    text = out.text_embeds.numpy()
    np.savez_compressed(out_dir / "embedding.npz", audio=audio, testo=text)
    meta = dict(model_id=model_id, revision=revision, sampling_rate=sr,
                files=[str(p.relative_to(PROJECT_ROOT)) for p in files],
                frasi=PHRASES, torch=str(torch.__version__), librosa=librosa.__version__)
    (out_dir / "provenienza.json").write_text(json.dumps(meta, indent=2))
    print(audio @ text.T, flush=True)
    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    prepare_clap_embeddings(args.output)
