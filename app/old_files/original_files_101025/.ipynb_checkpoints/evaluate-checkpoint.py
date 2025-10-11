# app/evaluate.py
import os, json, time, statistics as stats, math
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
# END


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


# ====== NEW: statistical test helpers ======
def mcnemar_exact_or_cc_p(b: int, c: int) -> float:
    """
    Two-sided McNemar p-value.
    - Exact binomial when (b+c) < 25
    - Chi-square with continuity correction otherwise.
    """
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        # exact binomial two-sided
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        p = min(1.0, 2.0 * tail)
        return float(p)
    # chi-square with CC, df=1; use survival func for chi2_1: erfc(sqrt(x/2))
    chi2 = (abs(b - c) - 1) ** 2 / float(n)
    p = math.erfc(math.sqrt(chi2 / 2.0))
    return float(p)

def bootstrap_macro_f1_diff(gold, base_pred, lora_pred, labels, B: int = 5000, seed: int = 42):
    """
    Paired bootstrap for ΔF1_macro = F1(LoRA) - F1(Base).
    Returns (delta_mean, ci_lo, ci_hi, p_two_sided).
    """
    rng = np.random.default_rng(seed)
    gold = np.asarray(gold); base_pred = np.asarray(base_pred); lora_pred = np.asarray(lora_pred)
    n = gold.size
    diffs = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        diffs[b] = (
            f1_score(gold[idx], lora_pred[idx], average="macro", labels=labels)
            - f1_score(gold[idx], base_pred[idx], average="macro", labels=labels)
        )
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(ci_lo), float(ci_hi), float(p)
# ====== END new helpers ======


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
# END


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

    # ====== NEW: save paired preds + run stats ======
    pairs = pd.DataFrame({
        "id": np.arange(len(lo["golds"])),
        "gold": lo["golds"],             # same order as zs["golds"]
        "base_pred": zs["preds"],
        "lora_pred": lo["preds"],
    })
    pairs.to_csv(mdir / "paired_preds.csv", index=False)

    # McNemar (paired accuracy)
    base_ok = pairs["base_pred"] == pairs["gold"]
    lora_ok = pairs["lora_pred"] == pairs["gold"]
    b = int((lora_ok & ~base_ok).sum())   # LoRA correct, Base wrong
    c = int((base_ok & ~lora_ok).sum())   # Base correct, LoRA wrong
    N = int(len(pairs))
    delta_acc = (b - c) / N
    p_mcnemar = mcnemar_exact_or_cc_p(b, c)

    # Paired bootstrap for macro-F1
    d_mean, ci_lo, ci_hi, p_boot = bootstrap_macro_f1_diff(
        pairs["gold"].to_numpy(),
        pairs["base_pred"].to_numpy(),
        pairs["lora_pred"].to_numpy(),
        labels=cfg.labels,
        B=5000,
        seed=123,
    )

    with open(mdir / "stat_tests.json", "w", encoding="utf-8") as f:
        json.dump({
            "mcnemar": {
                "b": b, "c": c, "N": N,
                "delta_accuracy": round(delta_acc, 6),
                "p": round(p_mcnemar, 6)
            },
            "bootstrap_macro_f1": {
                "delta_macro_f1": round(float(lo["macro_f1"] - zs["macro_f1"]), 6),
                "delta_mean_boot": round(d_mean, 6),
                "ci95": [round(ci_lo, 6), round(ci_hi, 6)],
                "p": round(p_boot, 6),
                "B": 5000
            }
        }, f, indent=2)

    print(
        f"[evaluate] Done.\n"
        f"ZeroShot macro-F1={rows[0]['macro_f1']}\n"
        f"LoRA macro-F1={rows[1]['macro_f1']} (r={best_r})\n"
        f"[stats] McNemar b={b} c={c} Δacc={delta_acc:.4f} p={p_mcnemar:.4g}\n"
        f"[stats] Δmacro-F1={lo['macro_f1']-zs['macro_f1']:.4f} "
        f"95%CI=[{ci_lo:.4f},{ci_hi:.4f}] p={p_boot:.4g}"
    )

if __name__ == "__main__":
    main()
