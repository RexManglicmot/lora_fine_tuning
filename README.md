## To LoRA and Beyond

**To LoRA and Beyond** is a production-grade data science project that fine-tunes large language models (LLMs) using **Low-Rank Adaptation (LoRA)** to automate tagging, classification, and routing of customer support messages. This project demonstrates how to adapt pretrained LLMs for domain-specific tasks using lightweight adapter tuning — without retraining the full model.

Leveraging LoRA adapters, the system supports scalable, pluggable inference by training separate adapters per product category or support domain. The architecture is designed for real-world deployment.

## Tech Stack

- Python · Hugging Face Transformers · PEFT (LoRA) · FastAPI · Docker · PyTorch · pandas · scikit-learn · pytest · Streamlit · GitHub Actions (CI) · logging · dotenv · modular code structure
