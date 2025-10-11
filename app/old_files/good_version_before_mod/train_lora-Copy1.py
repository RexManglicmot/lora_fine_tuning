# app/train_lora.py
import json, time, os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from app.config import cfg


os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- tiny helpers ----
def _dtype():
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.model.dtype, torch.float32)

def _read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def _prompt(tok, rec):
    msgs=[{"role":"system","content":rec["system"]},{"role":"user","content":rec["user"]}]
    if hasattr(tok,"apply_chat_template"):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"System: {rec['system']}\nUser: {rec['user']}\nAssistant: "

# added 9/30
def _encode(tok, rec, maxlen):
    pre = _prompt(tok, rec)
    # leave at least 1 token for the label; hard-truncate the prompt
    pre_ids = tok(pre, add_special_tokens=False, truncation=True,
                  max_length=maxlen - 1)["input_ids"]
    room = max(1, maxlen - len(pre_ids))  # never negative
    tgt_ids = tok(rec["assistant"], add_special_tokens=False,
                  truncation=True, max_length=room)["input_ids"]

    ids = (pre_ids + tgt_ids)[:maxlen]
    # mask prompt tokens; keep only the label part as targets
    labels = ([-100] * len(pre_ids)) + tgt_ids[:maxlen - len(pre_ids)]
    return torch.tensor(ids), torch.tensor(labels)
#end


def _collate(pad_id):
    def fn(batch):
        m=max(len(b[0]) for b in batch)
        def pad(x,val): return torch.cat([x, torch.full((m-len(x),), val, dtype=x.dtype)])
        inp=torch.stack([pad(b[0], pad_id) for b in batch])
        lab=torch.stack([pad(b[1], -100)  for b in batch])
        att=torch.stack([torch.cat([torch.ones(len(b[0]),dtype=torch.long),
                                    torch.zeros(m-len(b[0]),dtype=torch.long)]) for b in batch])
        return {"input_ids":inp,"labels":lab,"attention_mask":att}
    return fn

def _extract_label(txt):
    for lab in cfg.labels:
        if lab in txt: return lab
    low=txt.lower()
    for lab in cfg.labels:
        if lab.split("_")[0].lower() in low: return lab
    return cfg.labels[0]

class SFTDataset(Dataset):
    def __init__(self, tok, rows, maxlen): self.tok=tok; self.rows=rows; self.maxlen=maxlen
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return _encode(self.tok, self.rows[i], self.maxlen)

# added 9/29
@torch.inference_mode()
def _val_macro_f1(model, tok, records, device):
    """
    Score each label by conditional log-likelihood and pick argmax.
    Uses the label STRING as the prediction (no tensors saved).
    """
    from sklearn.metrics import f1_score
    preds, golds = [], []
    for r in records:
        pre = _prompt(tok, r)
        pre_ids = tok(
            pre, add_special_tokens=False, truncation=True,
            max_length=cfg.train.seq_len - 8
        )["input_ids"]

        best_label, best_score = None, -1e30
        for lab in cfg.labels:
            lab_ids = tok(lab, add_special_tokens=False)["input_ids"][:8]
            inp   = torch.tensor(pre_ids + lab_ids, device=device).unsqueeze(0)
            lab_t = torch.tensor([-100]*len(pre_ids) + lab_ids, device=device).unsqueeze(0)
            out = model(input_ids=inp, labels=lab_t)
            # higher is better: negate loss and scale by length of label
            score = -out.loss.item() * len(lab_ids)
            if score > best_score:
                best_score = score
                best_label = lab

        preds.append(best_label)
        golds.append(r["assistant"])

    return f1_score(golds, preds, average="macro", labels=cfg.labels)
# END

# @torch.inference_mode()
# def _val_macro_f1(model, tok, records, device):
#     preds,golds=[],[]
#     gk=dict(max_new_tokens=cfg.eval.max_new_tokens, temperature=cfg.eval.temperature,
#             do_sample=False, pad_token_id=tok.eos_token_id)
#     for r in records:
#         pre=_prompt(tok,r)

#         enc = tok(pre, return_tensors="pt", add_special_tokens=False,
#           truncation=True, max_length=cfg.train.seq_len - 1).to(device)
        
#         out=model.generate(**enc, **gk)[0]
#         gen=tok.decode(out[enc["input_ids"].shape[1]:], skip_special_tokens=True)
#         preds.append(_extract_label(gen)); golds.append(r["assistant"])
#     return f1_score(golds, preds, average="macro", labels=cfg.labels)

# ---- train ----
def main():
    # GPU maximize (safe defaults)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    device=torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    tok=AutoTokenizer.from_pretrained(cfg.model.primary)
    if tok.pad_token is None: tok.pad_token=tok.eos_token

    # was torch_dtype=_dtype() but changed to
    base=AutoModelForCausalLM.from_pretrained(
        cfg.model.primary, dtype=_dtype(), device_map=None
    )
    base.config.use_cache = False  # needed for grad checkpointing
    try: base.gradient_checkpointing_enable()
    except Exception: pass

    train=_read_jsonl(Path(cfg.paths.splits_dir)/"train.jsonl")
    val  =_read_jsonl(Path(cfg.paths.splits_dir)/"val.jsonl")
    ds=SFTDataset(tok, train, cfg.train.seq_len)

    #ADDED 9/29
    # --- class-balanced sampler (fix macro-F1 bias) ---
    lbl_counts = {lab: 0 for lab in cfg.labels}
    for r in train:
        lbl_counts[r["assistant"]] += 1

    # weight = 1 / count(label) for each training example
    weights = torch.tensor([1.0 / lbl_counts[r["assistant"]] for r in train], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    bs = max(1, cfg.train.batch_tokens // max(256, cfg.train.seq_len))
    dl = DataLoader(
        ds,
        batch_size=bs,
        sampler=sampler,          # <-- use sampler (do NOT set shuffle=True)
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate(tok.pad_token_id),
        persistent_workers=False
    )
    #END

    ranks = cfg.lora.r if isinstance(cfg.lora.r, list) else [cfg.lora.r]
    best_r, best_f1 = None, -1.0
    mdir=Path(cfg.paths.metrics_dir); mdir.mkdir(parents=True, exist_ok=True)

    amp_dtype=_dtype()
    use_amp = amp_dtype in (torch.float16, torch.bfloat16)

    # added 9/29
    scaler = torch.amp.GradScaler(enabled=(amp_dtype is torch.float16 and torch.cuda.is_available()))
    #coommented out on 9/29
    # scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype is torch.float16 and torch.cuda.is_available()))
    

    for r in map(int, ranks):
        # fresh LoRA head on the same base
        model = base
        lcfg=LoraConfig(r=r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                        target_modules=cfg.lora.target_modules, bias="none", task_type="CAUSAL_LM")
        model=get_peft_model(model, lcfg).to(device)

        # trainable params (MB)
        n_tr=sum(p.numel() for p in model.parameters() if p.requires_grad)
        bytes_per=2 if amp_dtype in (torch.float16, torch.bfloat16) else 4
        trainable_mb=n_tr*bytes_per/1e6

        # fused AdamW if available
        try:
            opt=torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, fused=True)
        except TypeError:
            opt=torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

        best_local=-1.0; curve=[]; wait=0; start=time.time()
        for ep in range(cfg.train.epochs):
            model.train(); acc=0
            for batch in tqdm(dl, desc=f"r={r} epoch {ep+1}/{cfg.train.epochs}", leave=False):
                batch={k:v.to(device, non_blocking=True) for k,v in batch.items()}
                if use_amp:
                    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
                        loss=model(**batch).loss / max(1, cfg.train.grad_accum)
                    if scaler.is_enabled():
                        scaler.scale(loss).backward(); acc+=1
                        if acc%cfg.train.grad_accum==0:
                            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                    else:
                        loss.backward(); acc+=1
                        if acc%cfg.train.grad_accum==0:
                            opt.step(); opt.zero_grad(set_to_none=True)
                else:
                    loss=model(**batch).loss / max(1, cfg.train.grad_accum)
                    loss.backward(); acc+=1
                    if acc%cfg.train.grad_accum==0:
                        opt.step(); opt.zero_grad(set_to_none=True)

            model.eval()
            with torch.no_grad():
                f1=_val_macro_f1(model, tok, val, device)
            curve.append(float(f1))

            if f1>best_local:
                best_local=f1; wait=0
                ck=Path(cfg.paths.checkpoints_dir)/f"lora_r{r}"
                ck.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ck); tok.save_pretrained(ck)
            else:
                wait+=1
                if wait>cfg.train.early_stop_patience: break

        wall=time.time()-start
        with open(mdir/f"lora_r{r}_train.json","w",encoding="utf-8") as f:
            json.dump({"r":r,
                       "trainable_params_mb":round(trainable_mb,3),
                       "training_time_wall":round(wall,2),
                       "val_macro_f1_curve":[round(x,4) for x in curve],
                       "best_val_macro_f1":round(best_local,4)}, f, indent=2)

        if best_local>best_f1:
            best_f1, best_r = best_local, r

    with open(mdir/"lora_best.json","w",encoding="utf-8") as f:
        json.dump({"best_r":best_r,"best_val_macro_f1":round(best_f1,4)}, f, indent=2)
    print(f"[train_lora] Done. Best r={best_r}  val macro-F1={best_f1:.4f}")

if __name__=="__main__":
    main()
