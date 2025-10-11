# app/utils.py
import json, torch
from app.config import cfg

# Map cfg.model.dtype → torch dtype; default fp32.
def _dtype():
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.model.dtype, torch.float32)

# Read JSONL into list[dict], skipping blank lines.
def _read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# Build a chat-style prompt; prefer tokenizer's chat template if available, otherwise fall back to a simple stitched string.
def _prompt(tok, rec):
    msgs=[{"role":"system","content":rec["system"]},{"role":"user","content":rec["user"]}]
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"System: {rec['system']}\nUser: {rec['user']}\nAssistant: "

# Heuristic parser: return a configured label if it appears directly or by prefix (before '_') in lowercased text; default to first label otherwise.
def _extract_label(txt):
    for lab in cfg.labels:
        if lab in txt: return lab
    low = txt.lower()
    for lab in cfg.labels:
        if lab.split("_")[0].lower() in low: return lab
    return cfg.labels[0]

# Run python3 -m app.utils
# No outputs, these are helper functions used in train.py and eval.py
# Was quick
# Next, is train.py