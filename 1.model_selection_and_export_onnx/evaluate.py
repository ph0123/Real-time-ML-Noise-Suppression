"""Evaluate one trained architecture on the VoiceBank-DEMAND test set.

Reports PESQ (wb) and STOI for noisy vs enhanced, plus the magnitude RMSE
between enhanced and clean spectra. plot_results.py calls evaluate_model() for all
five architectures; run standalone with MODEL=<name>.

Config via environment variables::

  MODEL=gru LIMIT=80 python evaluate.py    # LIMIT=0 -> full test set
"""

import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

from dataset import DATA_DIR
from model import apply_mask_istft, build_model, stft_mag

SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = int(os.environ.get("LIMIT", 0))  # 0 = full test set

HERE = Path(__file__).resolve().parent


@torch.no_grad()
def enhance(model, noisy):
    wave = torch.from_numpy(noisy).unsqueeze(0).to(DEVICE)
    mag, spec = stft_mag(wave)
    mask, _ = model(torch.log1p(mag))
    est = apply_mask_istft(spec, mask, wave.shape[-1])
    return est.squeeze(0).cpu().numpy()


def _spec_rmse(a, b):
    ma, _ = stft_mag(torch.from_numpy(a).unsqueeze(0))
    mb, _ = stft_mag(torch.from_numpy(b).unsqueeze(0))
    n = min(ma.shape[1], mb.shape[1])
    return math.sqrt(torch.mean((ma[:, :n] - mb[:, :n]) ** 2).item())


def evaluate_model(name: str, limit: int = LIMIT) -> dict:
    """Evaluate one architecture; return a dict of averaged metrics."""
    model = build_model(name).to(DEVICE)
    ckpt = HERE / f"checkpoint_{name}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    files = sorted((DATA_DIR / "noisy_test").glob("*.wav"))
    if limit:
        files = files[:limit]

    acc = {"pesq_noisy": [], "pesq_enh": [], "stoi_noisy": [], "stoi_enh": [],
           "rmse_noisy": [], "rmse_enh": []}
    for noisy_path in files:
        clean, _ = sf.read(DATA_DIR / "clean_test" / noisy_path.name, dtype="float32")
        noisy, _ = sf.read(noisy_path, dtype="float32")
        n = min(len(clean), len(noisy))
        clean, noisy = clean[:n], noisy[:n]
        enh = enhance(model, noisy)[:n]

        try:
            acc["pesq_noisy"].append(pesq(SR, clean, noisy, "wb"))
            acc["pesq_enh"].append(pesq(SR, clean, enh, "wb"))
        except Exception as e:  # pesq can fail on degenerate frames
            print(f"[{name}] pesq skipped {noisy_path.name}: {e}")
        acc["stoi_noisy"].append(stoi(clean, noisy, SR))
        acc["stoi_enh"].append(stoi(clean, enh, SR))
        acc["rmse_noisy"].append(_spec_rmse(noisy, clean))
        acc["rmse_enh"].append(_spec_rmse(enh, clean))

    metrics = {k: float(np.mean(v)) for k, v in acc.items() if v}
    metrics["n_files"] = len(files)
    print(f"[{name}] " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items()
                                  if k != "n_files"))
    return metrics


if __name__ == "__main__":
    name = os.environ.get("MODEL", "gru")
    m = evaluate_model(name)
    (HERE / f"eval_{name}.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in m.items()) + "\n")
