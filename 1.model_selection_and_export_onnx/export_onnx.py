"""Export a trained model as a streaming ONNX graph.

The graph runs ONE STFT frame at a time with a single GRU/CNN state tensor
passed in and out, so the C++ engine runs it causally in a real-time loop.
Output goes to the repo-root ``models/denoiser.onnx``.
Model selection (in priority order):
  MODEL=<name> env var, else comparison.json's "best_deployable", else "gru".
Only causal/streamable models can be exported (BiRNN is non-causal).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from model import (STREAMABLE, StreamingWrapper, build_model, streaming_state)

HERE = Path(__file__).resolve().parent
ONNX_PATH = HERE.parent / "models" / "denoiser.onnx"
COMPARISON = HERE / "comparison.json"


def pick_model() -> str:
    if os.environ.get("MODEL"):
        return os.environ["MODEL"].lower()
    if COMPARISON.exists():
        return json.loads(COMPARISON.read_text())["best_deployable"]
    return "gru"


def main():
    name = pick_model()
    if name not in STREAMABLE:
        raise SystemExit(f"'{name}' is not causal/streamable; cannot export.")

    model = build_model(name)
    model.load_state_dict(torch.load(HERE / f"checkpoint_{name}.pt",
                                     map_location="cpu"))
    model.eval()

    wrapper = StreamingWrapper(model)
    dummy_frame = torch.zeros(1, 1, model.norm.normalized_shape[0])  # (1,1,257)
    dummy_state = streaming_state(model)

    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (dummy_frame, dummy_state), str(ONNX_PATH),
        input_names=["log_mag", "state_in"],
        output_names=["mask", "state_out"],
        opset_version=17, dynamo=False,
    )
    print(f"Exported {name} -> {ONNX_PATH} (state shape {tuple(dummy_state.shape)})")

    # Sanity check: PyTorch vs ONNX Runtime must match.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH))
    f = np.random.rand(*dummy_frame.shape).astype(np.float32)
    s = dummy_state.numpy()
    m_onnx, _ = sess.run(None, {"log_mag": f, "state_in": s})
    with torch.no_grad():
        m_torch, _ = wrapper(torch.from_numpy(f), torch.from_numpy(s))
    print("max diff:", np.abs(m_onnx - m_torch.numpy()).max())  # ~1e-6


if __name__ == "__main__":
    main()
