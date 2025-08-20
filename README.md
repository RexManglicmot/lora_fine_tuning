## Staus

Still working on

To do: 
- create info the new dataset. Have a current dataset. 
- 


## Inspiration for this project




## Introduction
This project demonstrates how LoRA (Low‑Rank Adaptation) fine‑tunes a single base language model into multiple specialized versions using lightweight adapters. Each adapter is trained for a specific subset of the dataset, enabling businesses to achieve domain‑specific accuracy without retraining the full model.

Goal is to develop a production‑ready pipeline to train, evaluate, and manage LoRA adapters, using derived category subsets of the dataset, and compare their performance and efficiency against a zero‑shot base model.

## LoRA Tuning
LoRA (Low‑Rank Adaptation) fine‑tunes a pretrained LLM by adding small trainable layers—adapters—instead of updating all model weights. This allows domain specialization without retraining the entire model, saving time, compute, and storage.

An adapter is a compact, domain‑specific module (e.g., electronics, clothing, home) that plugs into the frozen base model to tailor it for a specific use case. 

In this project, we train three separate adapters: one for electronics, one for clothing, and one for home products. For example, the electronics adapter learns terms like "Bluetooth connectivity issues," while the clothing adapter understands "fabric shrinkage after wash." Once trained, adapters can be swapped in to provide targeted classification or domain‑aware support responses.

Business case: For companies with diverse product lines, adapters enable rapid customization for each market segment without retraining the entire model. This approach lowers operational costs, accelerates deployment, and improves customer satisfaction through domain‑specific accuracy. Businesses can prioritize critical domains first and expand coverage incrementally.


## Dataset
The dataset is from Kaggle and consist 
Dang, I'm doing something wrong. 

## Metrics
Each adapter is evaluated on its own filtered test set and compared to a zero‑shot base model.

Performance Metrics:

Accuracy – Percentage of correct predictions.
Precision – Percentage of predicted positives that are correct.
Recall – Percentage of actual positives correctly identified.
Macro‑F1 – Average F1 across all classes equally.
Micro‑F1 – F1 weighted by class frequency.
Confusion Matrix – Predicted vs. actual classification counts.

Efficiency Metrics (PEFT):

Adapter size – Storage size of LoRA adapter.
Training time – Duration of fine‑tuning.
Inference speed – Latency and throughput per prediction.


## Tech Stack


## Next steps
