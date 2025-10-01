### Results
| Method | Macro-F1 | Accuracy | F1 (Colon) | F1 (Lung) | Latency p50 / p95 (ms) | Tokens/s | Trainable Params (MB) | Train Time (min) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ZeroShot | 0.666 | 0.667 | 0.647 | 0.684 | 130.3 / 130.4 | 31.68 | 0.0 | 0.0 |
| LoRA-r8 | 0.954 | 0.958 | 0.939 | 0.968 | 172.9 / 173.1 | 17.36 | 41.9 | 8.4 |

### Statistical Tests
| Test | Metric | Effect (LoRA − Base) | 95% CI | p-value | Notes |
|---|---|---:|---|---:|---|
| Paired bootstrap | Macro-F1 | **+0.2880** | **[0.1795, 0.4026]** | **< 0.0002** | 5000 resamples |
| McNemar (paired) | Accuracy | **+0.2917** | — | **6e-06** | b=22, c=1 |
