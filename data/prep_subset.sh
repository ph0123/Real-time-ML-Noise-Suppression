#!/usr/bin/env bash
# Fast MATCHED SUBSET prep for a quick CPU demo train run.
#
# Extraction of the (many, tiny) wavs is fast; the slow part is the per-file
# ffmpeg resample, so we resample only the first N files of each train package.
# The full set is still available via:  bash data/prepare_data.sh train
#
# Usage:  bash data/prep_subset.sh [N]      (default N=1500)
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
N="${1:-1500}"

mkdir -p "$DATA_DIR/clean_train" "$DATA_DIR/noisy_train"

for entry in clean_trainset_28spk_wav:clean_train noisy_trainset_28spk_wav:noisy_train; do
  src="${entry%%:*}"; dst="$DATA_DIR/${entry##*:}"; zip="$DATA_DIR/${src}.zip"

  if [ ! -f "$zip" ]; then
    echo ">> downloading ${src}.zip"
    wget -q -O "$zip" "${BASE}/${src}.zip"
  fi

  echo ">> extracting ${src}.zip"
  rm -rf "$DATA_DIR/${src}"
  unzip -q "$zip" -d "$DATA_DIR"

  echo ">> resampling first ${N} files of ${src} -> ${dst}"
  i=0
  for f in $(ls "$DATA_DIR/${src}"/*.wav | sort | head -n "$N"); do
    ffmpeg -loglevel error -y -i "$f" -ar 16000 -ac 1 "$dst/$(basename "$f")"
    i=$((i + 1))
  done
  rm -rf "$DATA_DIR/${src}"
  echo ">> done ${dst}: $(ls "$dst" | wc -l) files"
done

echo "SUBSET PREP DONE"
