# app/evaluate.py
import os, json, time, statistics as stats
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from app.config import cfg

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# added 9/29
VERBALIZERS = {
    "Colon_Cancer":   ["Colon cancer",   "colorectal cancer"],
    "Lung_Cancer":    ["Lung cancer",    "pulmonary cancer"],
}
#END




# ---------- tiny helpers ----------
def _dtype():
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.model.dtype, torch.float32)

def _read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def _prompt(tok, rec):
    msgs=[{"role":"system","content":rec["system"]},{"role":"user","content":rec["user"]}]
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"System: {rec['system']}\nUser: {rec['user']}\nAssistant: "

def _extract_label(txt):
    for lab in cfg.labels:
        if lab in txt: return lab
    low = txt.lower()
    for lab in cfg.labels:
        if lab.split("_")[0].lower() in low: return lab
    return cfg.labels[0]

# added 9/29
@torch.inference_mode()
def _run_eval(model, tok, records, device):
    import time, statistics as stats, numpy as np
    from sklearn.metrics import f1_score, accuracy_score

    preds, golds, lat_ms = [], [], []
    total_label_tokens = 0

    for r in records:
        pre = _prompt(tok, r)
        pre_ids = tok(pre, add_special_tokens=False, truncation=True,
                      max_length=cfg.train.seq_len - 8)["input_ids"]

        t0 = time.perf_counter()
        best_label, best_score, best_v = None, -1e30, None
        for canon, verb_list in VERBALIZERS.items():
            best_vscore = -1e30
            for v in verb_list:
                v_ids = tok(v, add_special_tokens=False)["input_ids"][:8]
                inp   = torch.tensor(pre_ids + v_ids, device=device).unsqueeze(0)
                lab_t = torch.tensor([-100]*len(pre_ids) + v_ids, device=device).unsqueeze(0)
                out = model(input_ids=inp, labels=lab_t)
                score = -out.loss.item() * len(v_ids)
                if score > best_vscore:
                    best_vscore = score
                    if best_vscore > best_score:
                        best_label, best_score, best_v = canon, best_vscore, v
        dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        total_label_tokens += len(tok(best_v, add_special_tokens=False)["input_ids"]) if best_v else 0
        preds.append(best_label); golds.append(r["assistant"])

    macro = f1_score(golds, preds, average="macro", labels=cfg.labels)
    acc   = accuracy_score(golds, preds)
    per_c = f1_score(golds, preds, average=None, labels=cfg.labels)
    return {
        "preds": preds,
        "golds": golds,
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "per_class_f1": {lab: float(v) for lab, v in zip(cfg.labels, per_c)},
        "latency_p50_p95_ms": (float(stats.median(lat_ms)), float(np.percentile(lat_ms, 95))),
        "tokens_per_sec": float((total_label_tokens / max(1e-6, sum(lat_ms)/1000.0))),
        "latencies_ms": lat_ms,
    }
#END


# added 9/29
# @torch.inference_mode()
# def _run_eval(model, tok, records, device):
#     """
#     Label scoring (no free generation). Also times each sample.
#     """
#     from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
#     import numpy as np, statistics as stats, time

#     preds, golds, lat_ms = [], [], []
#     total_label_tokens = 0

#     for r in records:
#         pre = _prompt(tok, r)
#         pre_ids = tok(pre, add_special_tokens=False, truncation=True,
#                       max_length=cfg.train.seq_len - 8)["input_ids"]
#         t0 = time.perf_counter()
#         best_lab, best_score = None, -1e30
#         for lab in cfg.labels:
#             lab_ids = tok(lab, add_special_tokens=False)["input_ids"][:8]
#             inp = torch.tensor(pre_ids + lab_ids, device=device).unsqueeze(0)
#             lab_t = torch.tensor([-100]*len(pre_ids) + lab_ids, device=device).unsqueeze(0)
#             out = model(input_ids=inp, labels=lab_t)
#             score = -out.loss.item() * len(lab_ids)
#             if score > best_score:
#                 best_score, best_lab = score, lab
#         dt = (time.perf_counter() - t0) * 1000.0
#         lat_ms.append(dt)
#         total_label_tokens += len(tok(best_lab, add_special_tokens=False)["input_ids"])
#         preds.append(best_lab); golds.append(r["assistant"])

#     macro = f1_score(golds, preds, average="macro", labels=cfg.labels)
#     acc   = accuracy_score(golds, preds)
#     per_c = f1_score(golds, preds, average=None, labels=cfg.labels)
#     return {
#         "preds": preds,
#         "golds": golds,
#         "macro_f1": float(macro),
#         "accuracy": float(acc),
#         "per_class_f1": {lab: float(v) for lab, v in zip(cfg.labels, per_c)},
#         "latency_p50_p95_ms": (float(stats.median(lat_ms)), float(np.percentile(lat_ms, 95))),
#         "tokens_per_sec": float((total_label_tokens / max(1e-6, sum(lat_ms)/1000.0))),
#         "latencies_ms": lat_ms,
#     }
# end

def _load_base_and_tok():
    tok = AutoTokenizer.from_pretrained(cfg.model.primary)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg.model.primary, dtype=_dtype(), device_map=None)
    base.eval()
    return base, tok

def _load_lora(base, ckpt_dir):
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()
    return model

# ---------- main ----------
def main():
    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    test_path = Path(cfg.paths.splits_dir) / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}; run app.prepare_data first.")
    test = _read_jsonl(test_path)

    base, tok = _load_base_and_tok()
    base.to(device)

    # Zero-shot baseline
    zs = _run_eval(base, tok, test, device)

    # LoRA (best rank)
    mdir = Path(cfg.paths.metrics_dir); mdir.mkdir(parents=True, exist_ok=True)
    best_meta = json.loads(Path(mdir / "lora_best.json").read_text(encoding="utf-8"))
    best_r = int(best_meta["best_r"])
    train_meta = json.loads(Path(mdir / f"lora_r{best_r}_train.json").read_text(encoding="utf-8"))

    lora_dir = Path(cfg.paths.checkpoints_dir) / f"lora_r{best_r}"
    lora = _load_lora(base, lora_dir).to(device)
    lo = _run_eval(lora, tok, test, device)

    # Confusion matrix (LoRA) normalized by true row
    cm = confusion_matrix(lo["golds"], lo["preds"], labels=cfg.labels, normalize="true")
    pd.DataFrame(cm, index=cfg.labels, columns=cfg.labels).to_csv(mdir / "confusion_lora.csv", index=True)

    # results.csv (two rows: ZeroShot, LoRA)
    rows = [
        {
            "method": "ZeroShot",
            "macro_f1": round(zs["macro_f1"], 4),
            "accuracy": round(zs["accuracy"], 4),
            "per_class_f1": json.dumps(zs["per_class_f1"]),
            "latency_p50_p95_ms": f"{zs['latency_p50_p95_ms'][0]:.1f},{zs['latency_p50_p95_ms'][1]:.1f}",
            "tokens_per_sec": round(zs["tokens_per_sec"], 2),
            "trainable_params_mb": 0.0,
            "training_time_wall": 0.0,
        },
        {
            "method": f"LoRA-r{best_r}",
            "macro_f1": round(lo["macro_f1"], 4),
            "accuracy": round(lo["accuracy"], 4),
            "per_class_f1": json.dumps(lo["per_class_f1"]),
            "latency_p50_p95_ms": f"{lo['latency_p50_p95_ms'][0]:.1f},{lo['latency_p50_p95_ms'][1]:.1f}",
            "tokens_per_sec": round(lo["tokens_per_sec"], 2),
            "trainable_params_mb": round(float(train_meta["trainable_params_mb"]), 3),
            "training_time_wall": round(float(train_meta["training_time_wall"]), 2),
        },
    ]
    pd.DataFrame(rows).to_csv(mdir / "results.csv", index=False)

    print("[evaluate] Done.",
          f"ZeroShot macro-F1={rows[0]['macro_f1']}",
          f"LoRA macro-F1={rows[1]['macro_f1']} (r={best_r})",
          sep="\n")

if __name__ == "__main__":
    main()
