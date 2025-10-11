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
from app.utils import _dtype, _read_jsonl, _prompt, _extract_label


# False tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Label verbalizers used for label-scoring evaluation (multiple surface forms per class)
VERBALIZERS = {
    "Colon_Cancer":   ["Colon cancer",   "colorectal cancer"],
    "Lung_Cancer":    ["Lung cancer",    "pulmonary cancer"],
}


# ----- statistical test helpers ------

# Two-sided McNemar p-value.
#  - Exact binomial when (b+c) < 25
#  - Chi-square with continuity correction otherwise.
def mcnemar_exact_or_cc_p(b: int, c: int) -> float:
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


# Paired bootstrap for ΔF1_macro = F1(LoRA) - F1(Base).
# Returns (delta_mean, ci_lo, ci_hi, p_two_sided).
def bootstrap_macro_f1_diff(gold, base_pred, lora_pred, labels, B: int = 5000, seed: int = 42):
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
    
    # two-sided p-value against 0 difference
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(diffs.mean()), float(ci_lo), float(ci_hi), float(p)


# added 9/29
@torch.inference_mode()
def _run_eval(model, tok, records, device):
    # Run label-scoring evaluation over records:
    #   - For each example, score each label's verbalizers by appending them to the prompt and using negative loss (length-adjusted) as a compatibility score.
    #   - Collect predicted label, gold label, and latency.
    #   - Return metrics, per-class F1, latency summary, and tokens/sec

    # Create empty lists and set variable
    preds, golds, lat_ms = [], [], []
    total_label_tokens = 0

    for r in records:
        # Build prompt 
        pre = _prompt(tok, r)
        
        # Keep space for candidate label tokens
        pre_ids = tok(pre, add_special_tokens=False, truncation=True,
                      max_length=cfg.train.seq_len - 8)["input_ids"]

        # Score all verbalizers; pick the best label by highest score
        # start latency timer (milliseconds computed later)
        t0 = time.perf_counter()
        
        # track winning (label, score, verbalizer)
        best_label, best_score, best_v = None, -1e30, None
        
        # try each canonical label and its surface forms
        for canon, verb_list in VERBALIZERS.items():
            best_vscore = -1e30
            
            for v in verb_list:
                # Encode verbalizer tokens (cap at 8 tokens for stability/consistency)
                v_ids = tok(v, add_special_tokens=False)["input_ids"][:8]
                inp   = torch.tensor(pre_ids + v_ids, device=device).unsqueeze(0)
                # Supervise only the label tokens; mask prompt tokens with -100
                lab_t = torch.tensor([-100]*len(pre_ids) + v_ids, device=device).unsqueeze(0)
                out = model(input_ids=inp, labels=lab_t)
                
                # Length-aware compatibility score: higher = better
                score = -out.loss.item() * len(v_ids)
                
                # Keep best surface form for this canonical label
                if score > best_vscore:
                    best_vscore = score
                    
                    # And if it beats the global best, update global winner
                    if best_vscore > best_score:
                        best_label, best_score, best_v = canon, best_vscore, v
        
        # Record latency in ms for this example
        dt = (time.perf_counter() - t0) * 1000.0
        lat_ms.append(dt)
        
        # Track how many label tokens were evaluated (for tokens/sec)
        total_label_tokens += len(tok(best_v, add_special_tokens=False)["input_ids"]) if best_v else 0
        
        # Save prediction and gold label
        preds.append(best_label); golds.append(r["assistant"])

    # Aggregate metrics
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


# Load base model + tokenizer; ensure pad token; set model to eval mode
def _load_base_and_tok():
    tok = AutoTokenizer.from_pretrained(cfg.model.primary)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Was dtype=_dtype(), but changed it to torch_dtype()=_dtype() because it caused error!!
    base = AutoModelForCausalLM.from_pretrained(cfg.model.primary, torch_dtype=_dtype(), device_map=None)
    base.eval()
    return base, tok

# Attach LoRA adapters from checkpoint to base model; eval mode.
def _load_lora(base, ckpt_dir):
    model = PeftModel.from_pretrained(base, ckpt_dir)
    model.eval()
    return model


# ---------- main ----------

def main():
    # Evaluate Zero-shot base vs. best-rank LoRA; save metrics, preds, and stats tests.
    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    
    # Load test split produced by prepare_data
    test_path = Path(cfg.paths.splits_dir) / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing {test_path}; run app.prepare_data first.")
    test = _read_jsonl(test_path)

    # Base model + tokenizer on device
    base, tok = _load_base_and_tok()
    base.to(device)

    # Zero-shot baseline metrics/preds
    zs = _run_eval(base, tok, test, device)

    # Load best LoRA rank (selected during training) and evaluate
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

    # results.csv: one row per method (ZeroShot, LoRA)
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

    # Save paired preds + run stats
    pairs = pd.DataFrame({
        "id": np.arange(len(lo["golds"])),
        "gold": lo["golds"],             # same order as zs["golds"]
        "base_pred": zs["preds"],
        "lora_pred": lo["preds"],
    })
    pairs.to_csv(mdir / "paired_preds.csv", index=False)

    # McNemar counts: b = LoRA fixes; c = LoRA regressions
    base_ok = pairs["base_pred"] == pairs["gold"]
    lora_ok = pairs["lora_pred"] == pairs["gold"]
    b = int((lora_ok & ~base_ok).sum())   # LoRA correct, Base wrong
    c = int((base_ok & ~lora_ok).sum())   # Base correct, LoRA wrong
    N = int(len(pairs))
    delta_acc = (b - c) / N
    p_mcnemar = mcnemar_exact_or_cc_p(b, c)

    # Paired bootstrap CI + p-value for macro-F1 difference
    d_mean, ci_lo, ci_hi, p_boot = bootstrap_macro_f1_diff(
        pairs["gold"].to_numpy(),
        pairs["base_pred"].to_numpy(),
        pairs["lora_pred"].to_numpy(),
        labels=cfg.labels,
        B=5000,
        seed=123,
    )

    # Statistical test summary
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

    # Console summary for quick inspection
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

# 10/10/25
# Start: 1:39 pm
# End: 1:40 pm
# Was very quick
# Next, is plots.py



