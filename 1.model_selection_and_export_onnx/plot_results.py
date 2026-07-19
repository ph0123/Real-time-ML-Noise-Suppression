"""Train and compare all six architectures (CNN-1D, CNN-2D, RNN, BiRNN, LSTM, GRU).

Pipeline:
  1. train each model with early stopping (train.py) -> per-epoch curves
  2. evaluate each on the test set (evaluate.py)      -> PESQ / STOI / RMSE
  3. plot per-epoch train/test RMSE and accuracy + a final test-metric bar chart
  4. pick the best real-time-deployable model and record it for ONNX export

Figures written to results/:
  train_rmse.png       training magnitude-RMSE per epoch (all models)
  test_rmse.png        test magnitude-RMSE per epoch (all models)
  train_accuracy.png   training IBM classification accuracy per epoch
  test_accuracy.png    test IBM classification accuracy per epoch
  model_comparison.png final test PESQ / STOI / RMSE bars

Raw numbers + the chosen model go to comparison.json.
Config via env: EPOCHS, PATIENCE, MAX_FILES, SEGMENT_SECONDS, BATCH, LIMIT.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from evaluate import evaluate_model
from model import DEPLOYABLE, MODEL_NAMES
from train import EPOCHS, train_model

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"    # figures live in 1.model_selection_and_export_onnx/results/
LIMIT = int(os.environ.get("LIMIT", 80))

LABELS = {"cnn1d": "CNN-1D", "cnn2d": "CNN-2D", "rnn": "RNN",
          "birnn": "BiRNN", "lstm": "LSTM", "gru": "GRU"}
COLORS = {"cnn1d": "#4C72B0", "cnn2d": "#64B5CD", "rnn": "#DD8452",
          "birnn": "#55A868", "lstm": "#C44E52", "gru": "#8172B3"}


def plot_curves(curves, key, ylabel, title, filename):
    plt.figure(figsize=(8, 5))
    for name in MODEL_NAMES:
        c = curves.get(name)
        if c and c.get(key):
            plt.plot(range(1, len(c[key]) + 1), c[key], marker="o",
                     label=LABELS[name], color=COLORS[name])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = RESULTS / filename
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"wrote {out}")


def plot_comparison(metrics):
    names = [n for n in MODEL_NAMES if n in metrics]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = [
        ("pesq_enh", "pesq_noisy", "PESQ (wb) - higher is better"),
        ("stoi_enh", "stoi_noisy", "STOI - higher is better"),
        ("rmse_enh", "rmse_noisy", "Spectral RMSE - lower is better"),
    ]
    for ax, (enh_k, noisy_k, title) in zip(axes, panels):
        vals = [metrics[n][enh_k] for n in names]
        ax.bar([LABELS[n] for n in names], vals,
               color=[COLORS[n] for n in names])
        baseline = metrics[names[0]][noisy_k]  # noisy baseline (same for all)
        ax.axhline(baseline, ls="--", color="gray",
                   label=f"noisy baseline ({baseline:.2f})")
        ax.set_title(title)
        ax.legend()
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Test-set comparison of 6 architectures", fontweight="bold")
    fig.tight_layout()
    out = RESULTS / "model_comparison.png"
    fig.savefig(out, dpi=140)
    plt.close()
    print(f"wrote {out}")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    curves, metrics = {}, {}

    for name in MODEL_NAMES:
        curves[name] = train_model(name, EPOCHS)
        metrics[name] = evaluate_model(name, LIMIT)

    plot_curves(curves, "train_rmse", "Training magnitude RMSE",
                "Training RMSE per epoch", "train_rmse.png")
    plot_curves(curves, "test_rmse", "Test magnitude RMSE",
                "Test RMSE per epoch", "test_rmse.png")
    plot_curves(curves, "train_acc", "Training accuracy (IBM)",
                "Training classification accuracy per epoch", "train_accuracy.png")
    plot_curves(curves, "test_acc", "Test accuracy (IBM)",
                "Test classification accuracy per epoch", "test_accuracy.png")
    plot_comparison(metrics)

    # Best overall (any architecture) and best real-time-deployable model, both
    # by enhanced PESQ. BiRNN is non-causal and the CNNs are not O(1)/frame, so
    # deployment is chosen among the recurrent trio only.
    by_pesq = lambda n: metrics[n]["pesq_enh"]
    best_overall = max(metrics, key=by_pesq)
    deployable = [n for n in metrics if n in DEPLOYABLE]
    best_deploy = max(deployable, key=by_pesq)

    result = {
        "epochs_cap": EPOCHS,
        "eval_files": LIMIT,
        "metrics": metrics,
        "curves": curves,
        "best_overall": best_overall,
        "best_deployable": best_deploy,
    }
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2))
    print(f"\nBest overall (PESQ): {best_overall} "
          f"({metrics[best_overall]['pesq_enh']:.3f})")
    print(f"Best deployable/streaming: {best_deploy} "
          f"({metrics[best_deploy]['pesq_enh']:.3f})")
    print("wrote comparison.json")


if __name__ == "__main__":
    main()
