### Results
| Method | Macro-F1 | Accuracy | F1 (Colon) | F1 (Lung) | Latency p50 / p95 (ms) | Tokens/s | Trainable Params (MB) | Train Time (min) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ZeroShot | 0.666 | 0.667 | 0.647 | 0.684 | 130.4 / 130.5 | 31.92 | 0.0 | 0.0 |
| LoRA-r8 | 0.884 | 0.889 | 0.862 | 0.907 | 150.7 / 150.8 | 19.90 | 41.9 | 6.2 |

### Statistical Tests
| Test | Metric | Effect (LoRA − Base) | 95% CI | p-value | Notes |
|---|---|---:|---|---:|---|
| Paired bootstrap | Macro-F1 | **+0.2189** | **[0.0972, 0.3442]** | **0.0004** | 5000 resamples |
| McNemar (paired) | Accuracy | **+0.2222** | — | **0.0015** | b=20, c=4 |
