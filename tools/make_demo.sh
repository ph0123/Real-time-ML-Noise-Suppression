#!/usr/bin/env bash
# Produce the demo audio + before/after spectrograms used in results.md / README:
#   - export the best real-time model to models/denoiser.onnx
#   - denoise the first test clip with the C++ engine
#   - render noisy vs denoised spectrograms with the DSP toolkit's script
## Usage: bash tools/make_demo.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ORT_DIR="${ORT_DIR:-$ROOT/2.model_deployment_with_onnx/onnxruntime-linux-x64-1.18.0}"
export LD_LIBRARY_PATH="$ORT_DIR/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source venv/bin/activate 2>/dev/null || true
RESULTS=1.model_selection_and_export_onnx/results
mkdir -p demo "$RESULTS"

# Best real-time model (from comparison.json) -> models/denoiser.onnx
python 1.model_selection_and_export_onnx/export_onnx.py

first=$(ls data/noisy_test | head -1)
echo "Demo clip: $first"
cp "data/noisy_test/$first" demo/noisy_sample.wav
./2.model_deployment_with_onnx/build/offline_denoise models/denoiser.onnx demo/noisy_sample.wav demo/denoised_sample.wav

# Reuse the toolkit's spectrogram renderer (same 512/128 STFT).
SPEC="2.model_deployment_with_onnx/lib/dsp-fundamentals-toolkit/scripts/plot_spectrogram.py"
python "$SPEC" demo/noisy_sample.wav    "$RESULTS/noisy_spectrogram.png"
python "$SPEC" demo/denoised_sample.wav "$RESULTS/denoised_spectrogram.png"
echo "wrote $RESULTS/noisy_spectrogram.png and denoised_spectrogram.png"
