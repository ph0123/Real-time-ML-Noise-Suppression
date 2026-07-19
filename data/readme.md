# Data — VoiceBank-DEMAND (16 kHz)

Speech-enhancement benchmark used for training and evaluation.

- Source: https://datashare.ed.ac.uk/items/6ed35425-bf14-4d2b-93a1-0a4984952757
  (originally https://datashare.ed.ac.uk/handle/10283/2791)
- The original files are 48 kHz; the scripts below resample to **16 kHz mono**
  (the sample rate the whole project is fixed to).

## Layout (produced by the scripts)

```
data/
├── clean_train/   noisy_train/     # training pairs (matched by filename)
├── clean_test/    noisy_test/      # test pairs (matched by filename)
└── *.zip                           # downloaded archives (git-ignored)
```

`clean_*/<name>.wav` is the clean reference for `noisy_*/<name>.wav`.

## Download & resample

From the repo root:

```bash
bash data/prepare_data.sh test     # ~150 MB : test sets only
bash data/prepare_data.sh train    # ~2 GB   : full train sets
bash data/prepare_data.sh all      # test + train

# Quick CPU demo: a matched 1500-file train subset (extract all, resample N):
bash data/prep_subset.sh 1500
```

Requirements: `wget`, `unzip`, `ffmpeg`.

Only the resampled wavs and the zips are produced here; everything under
`data/` except this README is git-ignored (see the repo `.gitignore`).
