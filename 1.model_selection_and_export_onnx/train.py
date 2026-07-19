"""Train one mask-estimator architecture on VoiceBank-DEMAND.

Trains a single model chosen by name; plot_results.py calls train_model() for all
six. Each epoch records BOTH regression and classification metrics, on the
train set AND a held-out test probe, and training uses **early stopping**:
  * rmse - magnitude RMSE between masked-noisy and clean spectra (regression).
  * acc  - agreement between the thresholded mask (> 0.5) and the Ideal Binary
           Mask target IBM = [clean_mag / noisy_mag > 0.5]; i.e. per bin, did we
           correctly classify "keep this bin" vs "suppress it" (classification).
Rows are appended to ``training/train_log_<name>.csv`` with columns
``epoch, train_loss, train_rmse, train_acc, test_rmse, test_acc``. The best
(lowest test_rmse) weights are saved to ``checkpoint_<name>.pt``.

Config via environment variables::
  MODEL=gru EPOCHS=20 PATIENCE=4 MIN_DELTA=1e-4 BATCH=16 MAX_FILES=0 TEST_PROBE=40 python train.py
"""

import csv
import math
import os
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import DATA_DIR, VoiceBankDataset
from model import apply_mask_istft, build_model, stft_mag

EPOCHS = int(os.environ.get("EPOCHS", 20))         # max epochs (early stop below)
PATIENCE = int(os.environ.get("PATIENCE", 4))      # early-stop patience
MIN_DELTA = float(os.environ.get("MIN_DELTA", 1e-4))
BATCH = int(os.environ.get("BATCH", 16))
LR = float(os.environ.get("LR", 3e-4))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
MAX_FILES = int(os.environ.get("MAX_FILES", 0))    # 0 = all training files
TEST_PROBE = int(os.environ.get("TEST_PROBE", 40))  # test files for per-epoch eval
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HERE = Path(__file__).resolve().parent
EPS = 1e-8

def ckpt_path(name):
    return HERE / f"checkpoint_{name}.pt"

def log_path(name):
    return HERE / f"train_log_{name}.csv"

def si_sdr_loss(est, ref, eps=1e-8):
    """Negative scale-invariant SDR - standard speech-enhancement loss."""
    ref = ref - ref.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)
    proj = (torch.sum(est * ref, -1, keepdim=True) /
            (torch.sum(ref ** 2, -1, keepdim=True) + eps)) * ref
    noise = est - proj
    ratio = torch.sum(proj ** 2, -1) / (torch.sum(noise ** 2, -1) + eps)
    return -10 * torch.log10(ratio + eps).mean()

def ibm_accuracy(mask, noisy_mag, clean_mag):
    """Fraction of T-F bins whose thresholded mask matches the Ideal Binary Mask.
    IBM[bin] = 1 if the clean speech dominates (clean/noisy > 0.5), else 0. The
    prediction is (mask > 0.5). This turns mask estimation into a per-bin
    keep/suppress classification and reports its accuracy.
    """
    target = (clean_mag / (noisy_mag + EPS) > 0.5).float()
    pred = (mask > 0.5).float()
    return (pred == target).float().mean().item()

def load_test_probe(n):
    """Precompute STFT tensors for the first n test files (used every epoch)."""
    files = sorted((DATA_DIR / "noisy_test").glob("*.wav"))[:n]
    probe = []
    for np_ in files:
        clean, _ = sf.read(DATA_DIR / "clean_test" / np_.name, dtype="float32")
        noisy, _ = sf.read(np_, dtype="float32")
        m = min(len(clean), len(noisy))
        noisy_mag, _ = stft_mag(torch.from_numpy(noisy[:m]).unsqueeze(0).to(DEVICE))
        clean_mag, _ = stft_mag(torch.from_numpy(clean[:m]).unsqueeze(0).to(DEVICE))
        probe.append((torch.log1p(noisy_mag), noisy_mag, clean_mag))
    return probe

@torch.no_grad()
def eval_probe(model, probe):
    """Return (test_rmse, test_acc) over the test probe."""
    model.eval()
    mse_sum = acc_sum = 0.0
    for log_noisy, noisy_mag, clean_mag in probe:
        mask, _ = model(log_noisy)
        mse_sum += F.mse_loss(mask * noisy_mag, clean_mag).item()
        acc_sum += ibm_accuracy(mask, noisy_mag, clean_mag)
    n = len(probe)
    return math.sqrt(mse_sum / n), acc_sum / n

def train_model(name, epochs=EPOCHS):
    """Train one architecture with early stopping.
    Returns per-epoch curves: {train_rmse, test_rmse, train_acc, test_acc}.
    """
    train_set = VoiceBankDataset(str(DATA_DIR / "noisy_train"),
                                 str(DATA_DIR / "clean_train"), train=True)
    if MAX_FILES and MAX_FILES < len(train_set):
        train_set = torch.utils.data.Subset(train_set, range(MAX_FILES))
    loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                        num_workers=NUM_WORKERS, drop_last=True)
    probe = load_test_probe(TEST_PROBE)
    model = build_model(name).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{name}] device={DEVICE} train={len(train_set)} probe={len(probe)} "
          f"params={n_params:,} max_epochs={epochs} patience={PATIENCE}")
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    with open(log_path(name), "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_rmse",
                                "train_acc", "test_rmse", "test_acc"])
    curves = {"train_rmse": [], "test_rmse": [], "train_acc": [], "test_acc": []}
    best_rmse, bad_epochs = float("inf"), 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        tot_loss = tot_mse = tot_acc = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(DEVICE), clean.to(DEVICE)
            noisy_mag, noisy_spec = stft_mag(noisy)
            clean_mag, _ = stft_mag(clean)

            mask, _ = model(torch.log1p(noisy_mag))
            loss_mag = F.mse_loss(mask * noisy_mag, clean_mag)
            est_wave = apply_mask_istft(noisy_spec, mask, clean.shape[-1])
            loss = loss_mag + 0.05 * si_sdr_loss(est_wave, clean)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            tot_loss += loss.item()
            tot_mse += loss_mag.item()
            tot_acc += ibm_accuracy(mask, noisy_mag, clean_mag)

        sched.step()
        nb = len(loader)
        train_rmse = math.sqrt(tot_mse / nb)
        train_acc = tot_acc / nb
        test_rmse, test_acc = eval_probe(model, probe)

        curves["train_rmse"].append(train_rmse)
        curves["test_rmse"].append(test_rmse)
        curves["train_acc"].append(train_acc)
        curves["test_acc"].append(test_acc)
        print(f"[{name}] epoch {epoch:2d}: loss={tot_loss/nb:.4f} "
              f"train_rmse={train_rmse:.4f} train_acc={train_acc:.3f} "
              f"test_rmse={test_rmse:.4f} test_acc={test_acc:.3f}")
        with open(log_path(name), "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tot_loss/nb:.6f}",
                                    f"{train_rmse:.6f}", f"{train_acc:.6f}",
                                    f"{test_rmse:.6f}", f"{test_acc:.6f}"])

        # Early stopping on test RMSE; keep the best weights.
        if test_rmse < best_rmse - MIN_DELTA:
            best_rmse, bad_epochs = test_rmse, 0
            torch.save(model.state_dict(), ckpt_path(name))
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"[{name}] early stop at epoch {epoch} "
                      f"(best test_rmse={best_rmse:.4f})")
                break

    if not ckpt_path(name).exists():  # safety: always leave a checkpoint
        torch.save(model.state_dict(), ckpt_path(name))
    secs = time.perf_counter() - t_start
    curves["seconds"] = round(secs, 1)
    curves["epochs_run"] = len(curves["train_rmse"])
    print(f"[{name}] best test_rmse={best_rmse:.4f} -> {ckpt_path(name).name} "
          f"| trained {curves['epochs_run']} epochs in {secs:.1f}s")
    return curves

if __name__ == "__main__":
    train_model(os.environ.get("MODEL", "gru"))
