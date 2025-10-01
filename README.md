# LoRA Fine Tuning
> **LoRA (r=8)** lifts zero-shot **Macro-F1/Accuracy from 0.666/0.667 to 0.954/0.958** while training only **~42 MB** of adapters in **~8.4 min.**

## Inspiration
I wanted to do a LoRA because I wanted another way to tune a model without occuring computational costs and money with respect to full-fine tuning. I found these interesting videos HERE, HERE, and HERE, which talks about what LoRA is. It piqued my interest and again wanted to apply it to my own interests in puublic helath and the broad 

## Introduction
**LoRA (Low-Rank Adaptation)** fine-tunes a big model by freezing the original weights and learning tiny **adapters**—small add-on modules that nudge the model for your task. The adapter’s **rank** 𝑟 is just a knob for how big/expressive that add-on is: higher 𝑟 = more capacity; lower 𝑟 = smaller and faster. Because only the adapters learn (not the whole model), you train far fewer parameters, use less memory, and finish much quicker.

From a business view, LoRA cuts **GPU cost** and **time-to-value**, makes MLOps lighter (adapters are **small files** you can version, A/B, and roll back), and enables **safe personalization**—one vetted base model with per-client or per-use-case adapters you can swap in without touching production weights. That means faster iteration, easier governance, and scalable customization without duplicating the entire model.

Talk abour r= {8, 16}

batch_tokens: 2048        # effective batch size in tokens, epochs = 8, and a lr: 0.0001 # 1e-4

Objective of this project is:...




## Dataset
This project uses a Kaggle cancer text dataset covering Lung, Colon, and Thyroid cancers [Kaggle](https://www.kaggle.com/datasets/iamtanmayshukla/medical-text-classification-using-nlp)
). To reduce inference cost and focus on the target task, Thyroid was removed and the problem was framed as binary classification (Lung vs Colon). The full corpus is roughly Lung `n=2180` and Colon `n=2580`. The subset used here is shown below, and any residual skew is addressed during training with a class-balanced sampler.

| Split | Lung_Cancer | Colon_Cancer | Total |
|---|---:|---:|---:|
| train | 361 | 208 | 569 |
| val | 45 | 26 | 71 |
| test | 46 | 26 | 72 |
| **Total** | **452** | **260** | **712** |

*Counts reflect the subset used in this repo after preprocessing. The original dataset is larger.*

## Model
**`mistralai/Mistral-7B-Instruct-v0.3`** was the base model from [HuggineFace](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). It’s a compact, open-weights LLM with strong accuracy per parameter, so you get competitive zero-shot performance and a great starting point for small adapters. Because it’s memory- and compute-efficient, it fits comfortably on a single GPU and is LoRA-friendly;  can train tiny adapters quickly without touching the base weights. It's broad ecosystem support (tokenizer, checkpoints, PEFT compatibility) make it a practical, production-ready choice for fast iteration and low-cost customization.

## Workflow




## Metrics

**Macro-F1** is the average of per-class F1 scores with equal weight, so it shows balanced performance even when classes are imbalanced.

**Accuracy** is the fraction of all test cases predicted correctly, which can look high if one class dominates.

**Latency (p50 and p95)** reports typical and tail response times per request, with p50 as the median and p95 capturing slower outliers.

**Tokens/s** measures inference throughput as tokens processed per second, where higher numbers mean faster serving.

**Trainable Params (MB**) is the size of the parameters actually updated during training, indicating how lightweight the adaptation is.

**Train Time (min)** is the wall-clock time to reach the best checkpoint, showing how quickly the adapter converges.


## Results: Table, Performance
| Method | Macro-F1 | Accuracy | F1 (Colon) | F1 (Lung) | Latency p50 / p95 (ms) | Tokens/s | Trainable Params (MB) | Train Time (min) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ZeroShot | 0.666 | 0.667 | 0.647 | 0.684 | 130.3 / 130.4 | 31.68 | 0.0 | 0.0 |
| LoRA-r8 | 0.954 | 0.958 | 0.939 | 0.968 | 172.9 / 173.1 | 17.36 | 41.9 | 8.4 |

LoRA-r8 delivers a substantial lift over zero-shot: Macro-F1/Accuracy improve from 0.666/0.667 to 0.954/0.958, with per-class F1 reaching 0.939 (Colon) and 0.968 (Lung). The gains are balanced across classes, with only a modest latency increase (p50 ~130→173 ms) and lower throughput (31.68→17.36 tokens/s), while training stayed lightweight (~41.9 MB adapters, ~8.4 min). Because the adapter is tiny and quick to train, routine per-site retraining to manage drift is feasible without heavy infrastructure.





## Results: Visuals
![Macro-F1](./outputs/figures/method_bars_macroF1_accuracy.png) LoRA (r=8) boosts performance from Macro-F1 0.666 / Accuracy 0.667 (zero-shot) to 0.954 / 0.958—about a +0.29 absolute gain on both metrics. The sizable Macro-F1 jump indicates the improvement is spread across classes rather than driven by one label. And because LoRA updates only a small adapter, this gain comes with minimal training overhead compared with full fine-tuning.

<br>
<br>

![LoRA Confusion](./outputs/figures/confusion_matrix_lora.png) Colon_Cancer is correctly predicted 88% of the time and misclassified as Lung_Cancer 12% of the time; Lung_Cancer is 100% correct with 0% misclassified as Colon.
High values on the diagonal indicate strong recall per class (perfect for Lung, near-perfect for Colon), and the only error type shown is Colon → Lung.

<br>
<br>

![Epoch](./outputs/figures/val_macro_f1_per_epoch.png)
Validation Macro-F1 rises sharply from 0.268 → 0.678 → 0.886 by epoch 3 and reaches ~1.000 at epoch 4, with a tiny dip at epoch 5 before returning to 1.000 at epoch 6. The curve plateaus after epoch 4, indicating the model has effectively converged and extra epochs add little.

<br>
<br>

![GB and Adapter](./outputs/figures/efficiency_panel.png)
This panel shows the cost of adaptation. On the left, LoRA-r8 trains only about 42 MB of parameters (adapters), while ZeroShot trains none—highlighting that LoRA updates a tiny add-on rather than the whole model. On the right, LoRA reaches its best checkpoint in about 8.4 minutes, giving a quick turnaround for experiments.
Because LoRA learns small adapters, they’re fast to retrain, easy to store/share, and can be swapped without touching the frozen base model.

In practice, this is why LoRA: you get rapid task adaptation at low compute and storage cost, making iteration and deployment much lighter.

## Results: Table, Statistical Significance
| Test | Metric | Effect (LoRA − Base) | 95% CI | p-value | Notes |
|---|---|---:|---|---:|---|
| Paired bootstrap | Macro-F1 | **+0.2880** | **[0.1795, 0.4026]** | **< 0.0002** | 5000 resamples |
| McNemar (paired) | Accuracy | **+0.2917** | — | **6e-06** | b=22, c=1 |

The paired bootstrap resamples the same test cases to estimate the Macro F1 improvement, which here is +0.2880 with 95% CI [0.1795, 0.4026] and p < 0.0002, indicating a large and reliable gain. McNemar’s test compares accuracy on the same items by counting disagreements, yielding +0.2917 with p = 6e-06, which confirms the advantage is very unlikely to be due to chance.

For this Lung vs Colon cancer task, LoRA fixes far more errors than it introduces b = 22 vs c = 1, so the benefit should be felt in day-to-day use. The interval suggests future runs on similar data should retain a meaningful Macro F1 lift between roughly +0.18 and +0.40, while remaining work should focus on the residual Colon to Lung confusion shown in the confusion matrix.



## Next Steps:
 - Integrate **human-in-the-loop evaluations**, where the model proposes Cancer type and a pathologist confirms or corrects, and every false negative is escalated for immediate review.
 
 - **Plan a pilot study** using LoRA adapters on a single workflow (e.g., tumor-registry pre-labeling or pathology report triage).

## Conclusion
This project shows that a lightweight LoRA adapter can reliably **specialize a 7B base model for Lung vs Colon cancer classification**. With r=8, performance rose from Macro-F1/Accuracy 0.666/0.667 to 0.954/0.958, and the confusion matrix indicates the only notable residual error is Colon → Lung. Training touched only ~42 MB of parameters and converged in ~8.4 min, preserving the frozen base while delivering a large improvement. These results justify moving to a focused pilot with human-in-the-loop review, slice monitoring, and adapter variants per site/service line.

# Tech Stack
Python, CUDA GPU, PyTorch, Transformers, LLM, scikit-learn, pandas, numpy, matplotlib.