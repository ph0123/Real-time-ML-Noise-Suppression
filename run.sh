#!/usr/bin/env bash
# End-to-end reproduction of results.md:
#   submodule -> python env -> data -> train+compare (6 models, early stopping)
#   -> export best model -> build C++ -> benchmark ONNX-in-C++ -> spectrograms.
#
# Usage:
#   bash run.sh            # quick demo (train subset + early stopping)  ~1 h CPU
#   FULL=1 bash run.sh     # full VoiceBank-DEMAND + longer training
#
# Requires: python3, ffmpeg, wget, unzip, cmake, g++ (see README).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

ORT_VERSION="${ORT_VERSION:-1.18.0}"
export ORT_DIR="${ORT_DIR:-$ROOT/2.model_deployment_with_onnx/onnxruntime-linux-x64-$ORT_VERSION}"

echo "==> 0/6  DSP toolkit submodule"
git submodule update --init --recursive

echo "==> 1/6  Python env + dependencies"
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv --without-pip
  curl -sS https://bootstrap.pypa.io/get-pip.py | ./venv/bin/python
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install -q --index-url https://download.pytorch.org/whl/cpu torch
pip install -q -r 1.model_selection_and_export_onnx/requirements.txt

echo "==> 2/6  Dataset (16 kHz)"
if [ "${FULL:-0}" = "1" ]; then
  bash data/prepare_data.sh all
else
  bash data/prepare_data.sh test
  bash data/prep_subset.sh 1500
fi

echo "==> 3/6  Train + compare all 6 models"
cd 1.model_selection_and_export_onnx
if [ "${FULL:-0}" = "1" ]; then
  EPOCHS=30 PATIENCE=6 MAX_FILES=0   SEGMENT_SECONDS=4.0 LIMIT=0  python plot_results.py
else
  EPOCHS=20 PATIENCE=4 MAX_FILES=400 SEGMENT_SECONDS=1.5 LIMIT=50 python plot_results.py
fi
cd "$ROOT"

echo "==> 4/6  Download ONNX Runtime + build C++ engine"
if [ ! -d "$ORT_DIR" ]; then
  wget -q "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/onnxruntime-linux-x64-$ORT_VERSION.tgz"
  tar xzf "onnxruntime-linux-x64-$ORT_VERSION.tgz" -C 2.model_deployment_with_onnx && rm "onnxruntime-linux-x64-$ORT_VERSION.tgz"
fi
cmake -S 2.model_deployment_with_onnx -B 2.model_deployment_with_onnx/build -DORT_DIR="$ORT_DIR" >/dev/null
cmake --build 2.model_deployment_with_onnx/build -j

echo "==> 5/6  Benchmark every model in the C++ / ONNX-Runtime engine"
bash tools/benchmark.sh

echo "==> 6/6  Demo audio + before/after spectrograms"
bash tools/make_demo.sh

echo
echo "Done. See results.md, 1.model_selection_and_export_onnx/results/*.png, 1.model_selection_and_export_onnx/results/benchmark.md, demo/*.wav"
