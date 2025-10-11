# app/train_lora.py
import json, time, os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from app.config import cfg
from app.utils import _dtype, _read_jsonl, _prompt, _extract_label


# Avoid tokenizer parallel warnings; deterministic behavior is easier to debug
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Various labels → multiple surface forms used for label scoring at eval time
VERBALIZERS = {
    "Colon_Cancer": ["Colon cancer", "colorectal cancer"],
    "Lung_Cancer":  ["Lung cancer",  "pulmonary cancer"],
}


# ---- tiny helpers ----

# Convert one SFT record → (input_ids, labels) for causal LM training
def _encode(tok, rec, maxlen):
    
    # render chat-style prompt text
    pre = _prompt(tok, rec)
    
    # Tokenize prompt, hard-truncate so there's room for ≥1 label token
    pre_ids = tok(pre, add_special_tokens=False, truncation=True,
                  max_length=maxlen - 1)["input_ids"]
    
    # Remaining room for the label; never below 1
    room = max(1, maxlen - len(pre_ids))  # never negative
    
    # Tokenize the gold label into at most `room` tokens
    tgt_ids = tok(rec["assistant"], add_special_tokens=False,
                  truncation=True, max_length=room)["input_ids"]

    # Concatenate prompt + label; clip to maxlen (defensive)
    ids = (pre_ids + tgt_ids)[:maxlen]
    
    # Create labels: mask prompt tokens as -100 (ignored by loss),
    # keep label tokens as-is so they contribute to the loss
    labels = ([-100] * len(pre_ids)) + tgt_ids[:maxlen - len(pre_ids)]
    
    # Return tensors ready for the model
    return torch.tensor(ids), torch.tensor(labels)
    
    
# Build a batch collate function that:
#   - Pads input_ids to the max length in the batch with `pad_id`
#   - Pads labels with -100 so padding is ignored by the loss
#   - Creates attention_mask (1 for real tokens, 0 for padding)
def _collate(pad_id):
 
    def fn(batch):
        # Find the longest sequence length among items in the batch
        m = max(len(b[0]) for b in batch)
        
        # Pad a 1D tensor x up to length m using value `val`
        def pad(x, val): return torch.cat([x, torch.full((m - len(x),), val, dtype=x.dtype)])
        
        # Pad and stack input_ids using the tokenizer's pad_id
        inp = torch.stack([pad(b[0], pad_id) for b in batch])
        
        # Stack padded labels (use -100 so loss ignores padding positions)
        lab = torch.stack([pad(b[1], -100)  for b in batch])
        
        # Build attention_mask: 1s for real tokens, 0s for padded tail
        att = torch.stack([torch.cat([torch.ones(len(b[0]), dtype=torch.long),
                                      torch.zeros(m - len(b[0]), dtype=torch.long)]) for b in batch])
                                      
        # return dict expected by HF models
        return {"input_ids": inp, "labels": lab, "attention_mask": att}
    
    # Return the collate function for use in DataLoader(collate_fn=_collate(...))
    return fn


class SFTDataset(Dataset):
    # Wraps JSONL training records into a PyTorch Dataset.
    # Each item returns (input_ids, labels) tensors via _encode(...).
    
    def __init__(self, tok, rows, maxlen): 
        self.tok = tok         # tokenizer used to encode samples
        self.rows = rows       # list of dicts: {"system","user","assistant"}
        self.maxlen = maxlen    # max sequence length for encoding
        
    def __len__(self): 
        return len(self.rows)   # total number of samples
        
    def __getitem__(self, i): 
        # Encode the i-th record into model-ready tensors
        return _encode(self.tok, self.rows[i], self.maxlen)


# added 9/29 — returns macro_f1, accuracy, mean_val_loss
# Validation with label scoring (uses VERBALIZERS).
# Returns: macro_f1, accuracy, mean_val_loss
@torch.inference_mode()
def _val_metrics(model, tok, records, device):
    
    # Create empty lists
    preds, golds = [], []
    
    # Set varibales equal to 0 intially
    val_loss_sum, n = 0.0, 0

    for r in records:
        # render the chat-style prompt for this example
        pre = _prompt(tok, r)
        
        # Tokenize the prompt into IDs
        pre_ids = tok(pre, add_special_tokens=False, truncation=True,
                      # reserve room for candidate label tokens
                      max_length=cfg.train.seq_len - 8)["input_ids"]

        # ---- prediction by label scoring ----
        
        # track the best (label, score) over all verbalizers
        best_label, best_score = None, -1e30
        
        # Loop over each canonical label and its list of surface forms (verbalizers)
        for canon, verb_list in VERBALIZERS.items(): 
        
            # Best score among this label's own verbalizers
            best_vscore = -1e30
            
            # Try each verbalizer phrase for this canonical label
            for v in verb_list:
                
                # Tokenize the verbalizer text and cap it to 8 tokens (keeps scoring comparable/stable)
                v_ids = tok(v, add_special_tokens=False)["input_ids"][:8]
                
                # Build model input: prompt tokens + candidate verbalizer tokens (batch size = 1)
                inp   = torch.tensor(pre_ids + v_ids, device=device).unsqueeze(0)
                
                # Build labels: ignore the prompt part with -100 (no loss), supervise only the verbalizer tokens
                lab_t = torch.tensor([-100]*len(pre_ids) + v_ids, device=device).unsqueeze(0)
                
                # Forward pass with labels to get a loss we can use as a compatibility score
                out = model(input_ids=inp, labels=lab_t)
                
                # Heuristic score: higher is better (negative loss × length to reduce short-token bias)
                score = -out.loss.item() * len(v_ids)
                if score > best_vscore:
                    best_vscore = score
                    
            # Keep the label whose best verbalizer scored highest
            if best_vscore > best_score:
                best_score, best_label = best_vscore, canon
        
        # predicted canonical label
        preds.append(best_label)
        
        # gold canonical label
        golds.append(r["assistant"])

        # ---- validation loss on the GOLD label (best verbalizer) ----
        gold = r["assistant"]
        best_gold_loss = None
        
        # score all gold verbalizers; take the easiest (lowest loss)
        for v in VERBALIZERS.get(gold, [gold]):
            v_ids = tok(v, add_special_tokens=False)["input_ids"][:8]
            inp   = torch.tensor(pre_ids + v_ids, device=device).unsqueeze(0)
            lab_t = torch.tensor([-100]*len(pre_ids) + v_ids, device=device).unsqueeze(0)
            loss  = model(input_ids=inp, labels=lab_t).loss.item()
            if (best_gold_loss is None) or (loss < best_gold_loss):
                best_gold_loss = loss
                
        # accumulate per-example gold loss        
        val_loss_sum += float(best_gold_loss); n += 1

    # Compute macro-F1 over configured label set, overall accuracy, and mean validation loss
    macro = f1_score(golds, preds, average="macro", labels=cfg.labels)
    acc   = accuracy_score(golds, preds)
    vloss = val_loss_sum / max(1, n)
    
    return float(macro), float(acc), float(vloss)
    

# ---- train ----

def main():
    # GPU maximize (safe defaults)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    # Pick CUDA device from config, but just in case!
    device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer; ensure there is a pad token (fallback to EOS for causal LMs).
    tok = AutoTokenizer.from_pretrained(cfg.model.primary)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # was torch_dtype=_dtype() but changed to dtype
    base = AutoModelForCausalLM.from_pretrained(
        cfg.model.primary, 
        # Changed to torch_dtype.....originally it was dtype=_dtype()
        torch_dtype=_dtype(), 
        device_map=None)
    
    # Disable KV-cache during training; some models require this off for grad checkpointing to work.
    base.config.use_cache = False  
    
    # Turn on gradient checkpointing (reduces memory by re-computing activations on backward).
    # Wrapped in try/except to avoid crashing if the model/backend doesn't support it.
    try: 
        base.gradient_checkpointing_enable()
    except Exception: 
        pass

    # Load prepared splits and wrap train set in a Dataset with max sequence length.
    train = _read_jsonl(Path(cfg.paths.splits_dir) / "train.jsonl")
    val   = _read_jsonl(Path(cfg.paths.splits_dir) / "val.jsonl")
    ds = SFTDataset(tok, train, cfg.train.seq_len)
    

    # --- Class-balanced sampler (fix macro-F1 bias) ---
    
    # Count examples per label to create inverse-frequency sampling weights so rarer classes are sampled more often (helps macro-F1).
    lbl_counts = {lab: 0 for lab in cfg.labels}         # init count per known label
    for r in train:
        lbl_counts[r["assistant"]] += 1

    # weight = 1 / count(label) for each training example; sample with replacement.
    weights = torch.tensor([1.0 / lbl_counts[r["assistant"]] for r in train], dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Batch size heuristic based on token budget and sequence length.
    bs = max(1, cfg.train.batch_tokens // max(256, cfg.train.seq_len))
    
    # Build a DataLoader that class-balances samples via a sampler, batches and pads them (with attention masks), and streams efficiently to GPU.
    dl = DataLoader(
        ds,
        batch_size=bs,
        sampler=sampler,          # use sampler (do NOT set shuffle=True)
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate(tok.pad_token_id),
        persistent_workers=False)
    
    # Support either a single rank or a list of ranks for LoRA.
    ranks = cfg.lora.r if isinstance(cfg.lora.r, list) else [cfg.lora.r]
    
    # track best rank and its F1
    best_r, best_f1 = None, -1.0
    
    mdir = Path(cfg.paths.metrics_dir); mdir.mkdir(parents=True, exist_ok=True)

    # AMP setup: use autocast for fp16/bf16; GradScaler is needed only for fp16 on CUDA.
    amp_dtype = _dtype()
    use_amp = amp_dtype in (torch.float16, torch.bfloat16)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype is torch.float16 and torch.cuda.is_available()))

    # For each candidate rank r, configure a LoRA adapter on the base model (targeting the specified modules) and move the adapted model to the device.
    for r in map(int, ranks):
        model = base
        lcfg = LoraConfig(
            r=r, 
            lora_alpha=cfg.lora.alpha, 
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules, 
            bias="none", 
            task_type="CAUSAL_LM")
            
        model = get_peft_model(model, lcfg).to(device)

        # Estimate memory footprint of trainable parameters (rough, dtype-dependent).
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bytes_per = 2 if amp_dtype in (torch.float16, torch.bfloat16) else 4
        trainable_mb = n_tr * bytes_per / 1e6

        # Prefer fused AdamW on supported GPUs; otherwise fall back to standard AdamW.
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, fused=True)
        except TypeError:
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)


        # ---- Epoch loop with extra curves ----
        
        # Track the best F1 seen for this rank, metric curves, patience counter, and wall-clock start
        best_local = -1.0
        curve, curve_acc, curve_vloss, train_loss_curve = [], [], [], []
        wait = 0
        start = time.time()

        for ep in range(cfg.train.epochs):
            model.train()                   # enable training mode (dropout, etc.)
            acc = 0                         # steps accumulated toward grad_accum
            epoch_loss, steps = 0.0, 0      # running loss and step counter

            for batch in tqdm(dl, desc=f"r={r} epoch {ep+1}/{cfg.train.epochs}", leave=False):
                # Move tensors to the selected device; enable non_blocking copies for speed.
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

                if use_amp:
                    # Mixed precision forward/backward (saves memory, speeds up on GPU).
                    with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=amp_dtype):
                        # Divide loss for gradient accumulation
                        loss = model(**batch).loss / max(1, cfg.train.grad_accum)
                    
                    # Track mean epoch loss (detach to float)
                    epoch_loss += float(loss.item())
                    
                    if scaler.is_enabled():
                        # Safe fp16 backprop using GradScaler to avoid underflow.
                        scaler.scale(loss).backward(); acc += 1
                        
                        if acc % cfg.train.grad_accum == 0:
                            scaler.step(opt)                    # optimizer step with scaled grads
                            scaler.update()                     # adjust scale dynamically
                            opt.zero_grad(set_to_none=True)     # free memory faster
                    else:
                        # bf16 or CPU-autocast: no scaler needed.
                        loss.backward(); acc += 1
                        if acc % cfg.train.grad_accum == 0:
                            opt.step() 
                            opt.zero_grad(set_to_none=True)
                else:
                    # Full-precision training (fp32) fallback.
                    loss = model(**batch).loss / max(1, cfg.train.grad_accum)
                    epoch_loss += float(loss.item())
                    loss.backward(); acc += 1
                    if acc % cfg.train.grad_accum == 0:
                        opt.step()
                        opt.zero_grad(set_to_none=True)

                # count gradient accumulation “micro-steps”
                steps += 1

            # Mean training loss for this epoch (for plotting/monitoring).
            train_loss_curve.append(epoch_loss / max(1, steps))


            # ---- Validation pass: compute metrics on the held-out set ----
            # validation (macro-F1, accuracy, val loss)
            
            # disable dropout, etc. for evaluation
            model.eval()
            
            # no gradients during eval
            with torch.no_grad():
                f1, acc_val, vloss = _val_metrics(model, tok, val, device)
            
            # Log metric curves for plotting/early-stopping PER EPOCH
            curve.append(float(f1))
            curve_acc.append(float(acc_val))
            curve_vloss.append(float(vloss))
            
            
            # ---- Checkpointing & early stopping ----
            # Save the best-performing epoch (by macro-F1) for this rank r.
            if f1 > best_local:
                best_local = f1
                wait = 0
                ck = Path(cfg.paths.checkpoints_dir) / f"lora_r{r}"
                ck.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(ck)
                tok.save_pretrained(ck)
            else:
                # No improvement: increase patience counter and early-stop if exceeded
                wait += 1
                if wait > cfg.train.early_stop_patience:
                    break

        # ---- Per-rank summary JSON (curves + timing) ----
        
        wall = time.time() - start
        with open(mdir / f"lora_r{r}_train.json", "w", encoding="utf-8") as f:
            json.dump({
                "r": r,
                "trainable_params_mb": round(trainable_mb, 3),
                "training_time_wall": round(wall, 2),
                "val_macro_f1_curve": [round(x, 4) for x in curve],          
                "val_accuracy_curve": [round(x, 4) for x in curve_acc],
                "val_loss_curve":     [round(x, 4) for x in curve_vloss],
                "train_loss_curve":   [round(x, 4) for x in train_loss_curve],
                "best_val_macro_f1":  round(best_local, 4)
            }, f, indent=2)

        # Track global best across all tested ranks.
        if best_local > best_f1:
            best_f1, best_r = best_local, r


    # ---- Final best-rank report ----
    with open(mdir / "lora_best.json", "w", encoding="utf-8") as f:
        json.dump({"best_r": best_r, "best_val_macro_f1": round(best_f1, 4)}, f, indent=2)
    print(f"[train_lora] Done. Best r={best_r}  val macro-F1={best_f1:.4f}")

if __name__ == "__main__":
    main()

# 10/10/25
# Run python3 -m app.train
# Start: 1:18 pm
# End: 1:31
# Took 13 mins
# Next, is eval.py
# Also, dtype() made a huge difference :/
