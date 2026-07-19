#!/usr/bin/env bash
# Benchmark every trained model in the C++ / ONNX-Runtime streaming engine and
# write a Markdown timing table to 1.model_selection_and_export_onnx/results/benchmark.md.
## Each streamable model is exported to models/denoiser.onnx and run through
# offline_denoise, which reports per-hop time and the real-time factor (RTF).
# The 8 ms budget = one hop of 16 kHz audio; RTF < 1 means it keeps up live.
# BiRNN is bidirectional (non-causal) so it cannot stream -> reported as N/A.
## Usage: bash tools/benchmark.sh [noisy.wav]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ORT_DIR="${ORT_DIR:-$ROOT/2.model_deployment_with_onnx/onnxruntime-linux-x64-1.18.0}"
export LD_LIBRARY_PATH="$ORT_DIR/lib:${LD_LIBRARY_PATH:-}"
# shellcheck disable=SC1091
source venv/bin/activate 2>/dev/null || true

WAV="${1:-data/noisy_test/p232_001.wav}"
OUT="1.model_selection_and_export_onnx/results/benchmark.md"
mkdir -p 1.model_selection_and_export_onnx/results

{
  echo "| Model | Params | ONNX size | Per-hop (ms) | RTF | Real-time (<1)? |"
  echo "|---|---|---|---|---|---|"
} > "$OUT"

for M in cnn1d cnn2d rnn lstm gru; do
  MODEL="$M" python 1.model_selection_and_export_onnx/export_onnx.py >/dev/null 2>&1 || { echo "export $M failed"; continue; }
  size=$(du -h models/denoiser.onnx | cut -f1)
  params=$(PYTHONPATH=1.model_selection_and_export_onnx python -c "from model import build_model; print(f'{sum(p.numel() for p in build_model(\"$M\").parameters()):,}')")
  log=$(./2.model_deployment_with_onnx/build/offline_denoise models/denoiser.onnx "$WAV" "/tmp/bench_$M.wav")
  hop=$(echo "$log" | awk '/Per-hop:/{printf "%.2f", $2}')
  rtf=$(echo "$log" | awk '/Real-time factor:/{printf "%.3f", $3}')
  rt=$(awk "BEGIN{print ($rtf<1)?\"yes\":\"NO\"}")
  echo "| $M | $params | $size | $hop | $rtf | $rt |" >> "$OUT"
  echo "  $M: per-hop ${hop} ms, RTF ${rtf} (real-time: $rt)"
done

birnn_params=$(PYTHONPATH=1.model_selection_and_export_onnx python -c "from model import build_model; print(f'{sum(p.numel() for p in build_model(\"birnn\").parameters()):,}')")
echo "| birnn | $birnn_params | - | - | - | non-causal (offline only) |" >> "$OUT"

echo "wrote $OUT"
cat "$OUT"
