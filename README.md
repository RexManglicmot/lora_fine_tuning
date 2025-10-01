## Inspiration



## Introduction



## Dataset




## Model





## Workflow




## Metrics





## Results: Table, Performance







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






## Next Steps:





## Conclusion





# Tech Stack