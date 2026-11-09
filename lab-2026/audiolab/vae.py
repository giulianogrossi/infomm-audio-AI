"""Lesson 5, VAE for audio: prepare and read the morphing materials (RAVE).

The model is RAVE (IRCAM), in the variant trained on spoken and sung voices:
the encoder brings the audio to a continuous latent of 11 numbers every 2048
samples (about 43 ms at 48 kHz); the decoder rebuilds the waveform.

The offline path reads the precomputed materials in
``lab-2026/materiali/05-vae``: reconstructions of two voices, the morphing
chain between their latents and an out-of-domain example. Regeneration
requires ``torch`` and the model file, downloadable from Hugging Face with
``download_model()``.

RAVE model licence: CC-BY-NC-4.0 (see the repository).
Citation: Intelligent Instruments Lab, rave-models, 2023, doi:10.57967/hf/1235;
RAVE: Caillon & Esling, "RAVE: A variational autoencoder for fast and
high-quality neural audio synthesis", arXiv:2111.05011.
"""

import json
import platform
from pathlib import Path

from .paths import LAB_DIR, MATERIALS_DIR, PROJECT_ROOT

MODEL_DIR = LAB_DIR / "modelli-rave"
OFFLINE_DIR = MATERIALS_DIR / "05-vae"

REPO = "Intelligent-Instruments-Lab/rave-models"
REVISION = "c25a03d625840c40cd3a48779ed72f2a1947d7b4"
MODEL_FILE = "voice-multi-b2048-r48000-z11.ts"
MODEL_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{MODEL_FILE}"
LICENSE = "CC-BY-NC-4.0"

SAMPLE_RATE = 48_000
LATENT_DIM = 11
BLOCK = 2048
MORPH_ALPHAS = [i / 8 for i in range(9)]

VOICE1_FILE = PROJECT_ROOT / "audio-file/speech1.wav"
VOICE2_FILE = PROJECT_ROOT / "audio-file/speech2.wav"
OUT_OF_DOMAIN_FILE = PROJECT_ROOT / "datasets/UrbanSound8K/audio/fold5/100032-3-0-0.mp3"


def download_model(destination=None):
    """Download the model ``.ts`` file from Hugging Face, when not already there.

    Returns the local path. Uses the commit pinned in ``REVISION``.

    Parameters
    ----------
    destination : str or pathlib.Path or None, default None
        Where to save the file (default: ``MODEL_DIR / MODEL_FILE``).

    Returns
    -------
    pathlib.Path
        The local path of the model file.

    Examples
    --------
    >>> al.download_model()
    """
    import urllib.request

    destination = Path(destination or (MODEL_DIR / MODEL_FILE))
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("Scarico", MODEL_URL)
    with urllib.request.urlopen(MODEL_URL) as response, \
            destination.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return destination


def _load_model(path):
    import torch

    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    return model


def _encode_decode(model, y):
    """Encode and reconstruct a signal (pads the last block when needed)."""
    import numpy as np
    import torch

    x = torch.from_numpy(y).reshape(1, 1, -1)
    padding = (-x.shape[-1]) % BLOCK
    if padding:
        x = torch.nn.functional.pad(x, (0, padding))
    with torch.no_grad():
        z = model.encode(x)
        if isinstance(z, (tuple, list)):
            z = z[0]
        out = model.decode(z)
        if isinstance(out, (tuple, list)):
            out = out[0]
    z = z.squeeze(0).detach().cpu().numpy()
    out = out.squeeze().detach().cpu().numpy().astype("float32")[: len(y)]
    return z, out


def _rms_envelope(y, hop=512):
    import numpy as np

    n = len(y) // hop
    return np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)


def _mel_db(y, sr=SAMPLE_RATE):
    import librosa

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=80)
    return librosa.power_to_db(S + 1e-10)


def _mel_cosine(a, b, sr=SAMPLE_RATE):
    import numpy as np

    A, B = _mel_db(a, sr), _mel_db(b, sr)
    k = min(A.shape[1], B.shape[1])
    A, B = A[:, :k] - A[:, :k].mean(), B[:, :k] - B[:, :k].mean()
    return float((A * B).sum() / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))


def mean_mfcc(y, sr=SAMPLE_RATE):
    """Mean MFCC vector of a signal, used as a compact timbre descriptor.

    Parameters
    ----------
    y : numpy.ndarray
        Mono audio samples.
    sr : int, default SAMPLE_RATE
        Sampling rate in Hz (48000 for the RAVE materials).

    Returns
    -------
    numpy.ndarray
        Vector of 13 MFCCs, averaged over time with a hop of 512 samples.

    Examples
    --------
    >>> y, sr = al.load_audio(al.AUDIO_DIR / "speech1.wav", sr=48000)
    >>> descriptor = al.mean_mfcc(y, sr)
    >>> descriptor.shape
    (13,)
    """
    import librosa

    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512).mean(axis=1)


def reconstruction_metrics(reference, reconstruction, sr=SAMPLE_RATE):
    """Compare two signals, taking into account the causal delay of the model.

    The signals are truncated to the shorter length, centred and aligned
    using the lag at the largest absolute cross-correlation.

    Parameters
    ----------
    reference : numpy.ndarray
        Original mono signal.
    reconstruction : numpy.ndarray
        Reconstructed mono signal, at the same sampling rate.
    sr : int, default SAMPLE_RATE
        Sampling rate of both signals in Hz (48000 for the RAVE materials).

    Returns
    -------
    dict
        "ritardo": estimated lag in samples (positive when the reconstruction
        is delayed); "inviluppo": correlation of RMS envelopes in blocks of
        512 samples after alignment; "mel": cosine similarity between centred
        log-Mel spectrograms with 80 bands after alignment.

    Notes
    -----
    Use signals long enough to compare several RMS blocks. Envelope
    correlation can be NaN for constant envelopes or too few blocks.

    Examples
    --------
    >>> metrics = al.reconstruction_metrics(originale, ricostruita, sr=48000)
    >>> print(metrics["ritardo"], metrics["inviluppo"], metrics["mel"])
    """
    import numpy as np
    from scipy.signal import correlate, correlation_lags

    n = min(len(reference), len(reconstruction))
    a = reference[:n] - reference[:n].mean()
    b = reconstruction[:n] - reconstruction[:n].mean()
    cc = correlate(b, a, mode="full", method="fft")
    lag = int(correlation_lags(len(b), len(a), mode="full")[int(np.argmax(np.abs(cc)))])
    if lag >= 0:
        b2, a2 = b[lag:], a[: len(b) - lag]
    else:
        a2, b2 = a[-lag:], b[: len(a) + lag]
    e1, e2 = _rms_envelope(a2), _rms_envelope(b2)
    k = min(len(e1), len(e2))
    envelope = float(np.corrcoef(e1[:k], e2[:k])[0, 1])
    return {"ritardo": lag, "inviluppo": envelope, "mel": _mel_cosine(a2, b2, sr)}


def _readme_text(meta):
    alpha_rows = ", ".join(f"{i} → {a:.3f}".replace(".", ",") for i, a in enumerate(MORPH_ALPHAS))
    return f"""# Materiali del laboratorio VAE (RAVE) — capitolo 5

Ricostruzioni e catena di morphing prodotte con RAVE, un autoencoder
variazionale per l'audio (IRCAM). Modello: `{meta['file']}` dal repository
`{meta['repo']}` (commit `{meta['revisione']}`), licenza {meta['licenza']}.
Generati il {meta['generato'][:10]} con `lab-2026/audiolab/vae.py` (torch {meta['torch']}).

## Contenuto

- `voce1.wav`, `voce2.wav`: le due clip di partenza (audio-file/speech1.wav e
  speech2.wav, ricampionate a {meta['sr']} Hz mono). Sono registrazioni reali del corso.
- `voce1_ricostruita.wav`, `voce2_ricostruita.wav`: le stesse clip dopo
  codifica e decodifica. La forma d'onda non coincide campione per campione
  (il decoder rigenera la fase); contenuto e inviluppo restano simili.
- `morph_0.wav` … `morph_8.wav`: interpolazione dei latenti fra voce1 e voce2:
  {alpha_rows}. In ascolto usare un guadagno comune a tutta la catena.
- `latenti.npz`: latenti `voce1`, `voce2` e la catena `morph` (forma 9×{meta['latente_dim']}×T).
- `fuori_dominio_originale.wav`, `fuori_dominio_ricostruito.wav`: un abbaio
  (UrbanSound8K) passato attraverso il modello voce: serve a mostrare che il
  modello è specializzato e degrada fuori dal suo dominio.
- `provenienza.json`: modello, revisione, sha256 del file, versioni, metriche.

## Note

- I file sono WAV FLOAT; non sono registrazioni: la ricostruzione è sintetica.
- I modelli RAVE dell'Intelligent Instruments Lab sono distribuiti con licenza
  CC-BY-NC-4.0: uso non commerciale, citare la fonte.
- Per rigenerare tutto (serve `torch` e il modello, ~150 MB):
  `python -m audiolab.vae` — oppure `al.vae.generate_materials()` da Python.
"""


def generate_materials(folder=None):
    """Regenerate the materials (reconstructions, morphing, out of domain, metadata).

    Requires ``torch``. Writes to ``lab-2026/materiali/05-vae`` unless told
    otherwise; downloads the model only when missing. Returns the folder
    written.

    Parameters
    ----------
    folder : str or pathlib.Path or None, default None
        Destination folder.

    Returns
    -------
    pathlib.Path
        The folder written.

    Examples
    --------
    >>> al.vae.generate_materials()
    """
    import hashlib
    import time
    from datetime import datetime, timezone
    from importlib.metadata import version

    import numpy as np
    import librosa
    import soundfile as sf
    import torch

    folder = Path(folder or OFFLINE_DIR)
    folder.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_FILE
    if not model_path.is_file():
        model_path = download_model()
    model = _load_model(model_path)

    def save(name, y):
        sf.write(folder / name, y, SAMPLE_RATE, subtype="FLOAT")

    def load_48(path):
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        return y.astype("float32")

    start = time.perf_counter()
    y1 = load_48(VOICE1_FILE); z1, o1 = _encode_decode(model, y1)
    y2 = load_48(VOICE2_FILE); z2, o2 = _encode_decode(model, y2)
    save("voce1.wav", y1); save("voce2.wav", y2)
    save("voce1_ricostruita.wav", o1); save("voce2_ricostruita.wav", o2)

    morphs, latent_chain = [], []
    for i, a in enumerate(MORPH_ALPHAS):
        z = ((1 - a) * z1 + a * z2).astype("float32")
        with torch.no_grad():
            o = model.decode(torch.from_numpy(z).unsqueeze(0).float())
            if isinstance(o, (tuple, list)):
                o = o[0]
        o = o.squeeze().detach().cpu().numpy().astype("float32")[: len(y1)]
        save(f"morph_{i}.wav", o)
        morphs.append(o); latent_chain.append(z)

    yf = load_48(OUT_OF_DOMAIN_FILE); zf, of = _encode_decode(model, yf)
    save("fuori_dominio_originale.wav", yf); save("fuori_dominio_ricostruito.wav", of)

    np.savez_compressed(folder / "latenti.npz",
                        voce1=z1, voce2=z2, morph=np.stack(latent_chain))

    mf1, mf2 = mean_mfcc(y1), mean_mfcc(y2)
    d_from_1 = [float(np.linalg.norm(mean_mfcc(o) - mf1)) for o in morphs]
    d_from_2 = [float(np.linalg.norm(mean_mfcc(o) - mf2)) for o in morphs]
    adjacent = [_mel_cosine(morphs[i], morphs[i + 1]) for i in range(len(morphs) - 1)]

    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    meta = {
        "repo": REPO, "file": MODEL_FILE, "revisione": REVISION,
        "licenza": LICENSE, "sha256_modello": sha,
        "sr": SAMPLE_RATE, "latente_dim": LATENT_DIM, "blocco": BLOCK, "causale": True,
        "alpha": MORPH_ALPHAS,
        "sorgenti": {
            "voce1": "audio-file/speech1.wav",
            "voce2": "audio-file/speech2.wav",
            "fuori_dominio": "datasets/UrbanSound8K/audio/fold5/100032-3-0-0.mp3",
        },
        "files": sorted(p.name for p in folder.glob("*.wav")),
        "metriche": {
            "ricostruzioni": {
                "voce1": reconstruction_metrics(y1, o1),
                "voce2": reconstruction_metrics(y2, o2),
                "fuori_dominio": reconstruction_metrics(yf, of),
            },
            "morph": {"d_mfcc_da_voce1": d_from_1, "d_mfcc_da_voce2": d_from_2,
                      "mel_adiacenti": adjacent},
        },
        "torch": version("torch"), "numpy": version("numpy"),
        "librosa": version("librosa"), "soundfile": version("soundfile"),
        "python": platform.python_version(), "macchina": platform.machine(),
        "script": "lab-2026/audiolab/vae.py",
        "generato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "secondi": round(time.perf_counter() - start, 1),
    }
    (folder / "provenienza.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    (folder / "LEGGIMI.md").write_text(_readme_text(meta))

    print(f"Materiali in {folder}")
    print("ricostruzioni:", {k: {kk: round(vv, 3) for kk, vv in v.items()}
                             for k, v in meta["metriche"]["ricostruzioni"].items()})
    print("dMFCC da voce1:", [round(v, 1) for v in d_from_1])
    print("dMFCC da voce2:", [round(v, 1) for v in d_from_2])
    print("somiglianza adiacenti:", [round(v, 3) for v in adjacent])
    return folder


def load_materials(folder=None):
    """Read the precomputed materials of the lab.

    Parameters
    ----------
    folder : str or pathlib.Path or None, default None
        The folder written by ``generate_materials()``.

    Returns
    -------
    dict
        With "voce1", "voce2", "ricostruite" (list), "morph" (list of
        (alpha, signal) pairs), "latenti" (dict of arrays), "fuori_dominio"
        (dict with "originale" and "ricostruito"), "sr", "alpha" and "meta".

    Examples
    --------
    >>> materials = al.vae.load_materials()
    >>> len(materials["morph"])
    9
    """
    import numpy as np
    import soundfile as sf

    folder = Path(folder or OFFLINE_DIR)
    meta = json.loads((folder / "provenienza.json").read_text())
    sr = int(meta["sr"])

    def read_wav(name):
        y, s = sf.read(folder / name, dtype="float32", always_2d=False)
        if s != sr or y.ndim != 1:
            raise ValueError(f"File inatteso nei materiali: {name} ({s} Hz, {y.ndim} canali)")
        return y

    voice1 = read_wav("voce1.wav"); voice2 = read_wav("voce2.wav")
    reconstructions = [read_wav("voce1_ricostruita.wav"), read_wav("voce2_ricostruita.wav")]
    alpha = [float(a) for a in meta["alpha"]]
    morph = [(a, read_wav(f"morph_{i}.wav")) for i, a in enumerate(alpha)]
    with np.load(folder / "latenti.npz") as archive:
        latents = {k: archive[k] for k in archive.files}
    out_of_domain = {
        "originale": read_wav("fuori_dominio_originale.wav"),
        "ricostruito": read_wav("fuori_dominio_ricostruito.wav"),
    }
    return {"voce1": voice1, "voce2": voice2, "ricostruite": reconstructions,
            "morph": morph, "latenti": latents, "fuori_dominio": out_of_domain,
            "sr": sr, "alpha": alpha, "meta": meta}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Materiali VAE (RAVE) per il laboratorio 5")
    parser.add_argument("--output", default=None,
                        help="cartella di destinazione (default: materiali/05-vae)")
    parser.add_argument("--download", action="store_true",
                        help="scarica solo il modello e termina")
    args = parser.parse_args()
    if args.download:
        print(download_model())
    else:
        generate_materials(args.output)
