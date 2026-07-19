"""Fast self-test of the model + streaming logic (no dataset needed).
Checks, for all five architectures:
  * forward() returns a mask of shape (B, T, 257) in [0, 1]
  * for causal models, the frame-by-frame StreamingWrapper matches the
    full-sequence forward after the warm-up (this is what the C++ engine relies
    on). BiRNN is non-causal and is only shape-checked.
Exits non-zero on failure, so CI can gate on it. Run: `python selftest.py`.
"""

import sys
import torch
from model import (MODEL_NAMES, STREAMABLE, StreamingWrapper, build_model, streaming_state)

WARMUP = 40  # frames; streaming is exact only after the receptive field fills

def main() -> int:
    torch.manual_seed(0)
    ok = True
    for name in MODEL_NAMES:
        model = build_model(name).eval()
        seq = torch.rand(1, 80, 257)
        with torch.no_grad():
            full, _ = model(seq)

        shape_ok = tuple(full.shape) == (1, 80, 257)
        range_ok = bool((full >= 0).all() and (full <= 1).all())
        line = f"{name:5s} params={sum(p.numel() for p in model.parameters()):>9,}"

        if name in STREAMABLE:
            wrap, state, outs = StreamingWrapper(model), streaming_state(model), []
            with torch.no_grad():
                for t in range(80):
                    y, state = wrap(seq[:, t:t + 1, :], state)
                    outs.append(y)
            diff = (full[:, WARMUP:] - torch.cat(outs, 1)[:, WARMUP:]).abs().max().item()
            stream_ok = diff < 1e-4
            line += f"  stream_diff={diff:.1e}"
        else:
            stream_ok = True
            line += "  (non-causal, comparison only)"

        passed = shape_ok and range_ok and stream_ok
        ok = ok and passed
        print(f"[{'OK ' if passed else 'FAIL'}] {line}")

    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
