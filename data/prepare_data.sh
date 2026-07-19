#!/usr/bin/env bash
# Download VoiceBank-DEMAND and resample it to 16 kHz mono into this data/ dir.
#
# Layout produced (relative to repo root):
#   data/clean_train  data/noisy_train  data/clean_test  data/noisy_test
#
# Usage:  bash data/prepare_data.sh [test|train|all]
#   test  -> only the (small) test sets     ~150 MB
#   train -> only the train sets            ~ 2 GB
#   all   -> test + train sets (default)    ~ 2-3 GB
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="https://datashare.ed.ac.uk/bitstream/handle/10283/2791"
MODE="${1:-all}"

# package_zip:dest_subdir
TEST_PKGS=(
  "clean_testset_wav:clean_test"
  "noisy_testset_wav:noisy_test"
)
TRAIN_PKGS=(
  "clean_trainset_28spk_wav:clean_train"
  "noisy_trainset_28spk_wav:noisy_train"
)

case "$MODE" in
  test)  pkgs=("${TEST_PKGS[@]}") ;;
  train) pkgs=("${TRAIN_PKGS[@]}") ;;
  all)   pkgs=("${TEST_PKGS[@]}" "${TRAIN_PKGS[@]}") ;;
  *) echo "Usage: $0 [test|train|all]"; exit 1 ;;
esac

mkdir -p "$DATA_DIR"/{clean_train,noisy_train,clean_test,noisy_test}

for entry in "${pkgs[@]}"; do
  src="${entry%%:*}"; dst="$DATA_DIR/${entry##*:}"
  zip="$DATA_DIR/${src}.zip"

  if [ ! -f "$zip" ]; then
    echo ">> downloading ${src}.zip"
    wget -q --show-progress -O "$zip" "${BASE}/${src}.zip"
  fi

  echo ">> extracting ${src}.zip"
  rm -rf "$DATA_DIR/${src}"
  unzip -q "$zip" -d "$DATA_DIR"

  echo ">> resampling ${src} -> 16 kHz mono into ${dst}"
  for f in "$DATA_DIR/${src}"/*.wav; do
    ffmpeg -loglevel error -y -i "$f" -ar 16000 -ac 1 "$dst/$(basename "$f")"
  done
  rm -rf "$DATA_DIR/${src}"
  echo ">> done ${dst}: $(ls "$dst" | wc -l) files"
done

echo "All done."
