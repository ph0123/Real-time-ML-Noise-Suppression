"""VoiceBank-DEMAND dataset: pairs of (noisy, clean) 16 kHz waveforms.

The resampled dataset is expected under the repo-root ``data/`` directory
(NOT inside ``training/`` and NOT next to the model), laid out as::
    data/
      clean_train/  noisy_train/
      clean_test/   noisy_test/
"""

import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

# Repo-root data directory, resolved independently of the current working dir.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

SR = 16000
# Random crop length used for training. Override with SEGMENT_SECONDS to trade quality for speed on CPU-only machines (e.g. 2.0 roughly halves per-batch cost).
SEGMENT_SECONDS = float(os.environ.get("SEGMENT_SECONDS", 4.0))

class VoiceBankDataset(Dataset):
    """Pairs of (noisy, clean) waveforms, matched by filename."""

    def __init__(self, noisy_dir: str, clean_dir: str, train: bool = True):
        self.noisy_files = sorted(Path(noisy_dir).glob("*.wav"))
        self.clean_dir = Path(clean_dir)
        self.train = train
        assert self.noisy_files, f"No wav files in {noisy_dir}"

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, idx):
        noisy_path = self.noisy_files[idx]
        clean_path = self.clean_dir / noisy_path.name

        noisy, sr1 = sf.read(noisy_path, dtype="float32")
        clean, sr2 = sf.read(clean_path, dtype="float32")
        assert sr1 == SR and sr2 == SR, "Resample dataset to 16 kHz first"

        n = min(len(noisy), len(clean))
        noisy, clean = noisy[:n], clean[:n]

        seg = int(SEGMENT_SECONDS * SR)
        if self.train:
            if n > seg:
                start = random.randint(0, n - seg)
                noisy, clean = noisy[start:start + seg], clean[start:start + seg]
            else:
                pad = seg - n
                noisy = np.pad(noisy, (0, pad))
                clean = np.pad(clean, (0, pad))
        return torch.from_numpy(noisy.copy()), torch.from_numpy(clean.copy())
