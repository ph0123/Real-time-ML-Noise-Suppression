| Model | Params | ONNX size | Per-hop (ms) | RTF | Real-time (<1)? |
|---|---|---|---|---|---|
| cnn1d | 231,043 | 912K | 0.72 | 0.090 | yes |
| cnn2d | 19,363 | 88K | 5.92 | 0.740 | yes |
| rnn | 329,987 | 1.3M | 0.15 | 0.019 | yes |
| lstm | 1,120,259 | 4.3M | 0.39 | 0.048 | yes |
| gru | 856,835 | 3.3M | 0.28 | 0.035 | yes |
| birnn | 790,275 | - | - | - | non-causal (offline only) |
