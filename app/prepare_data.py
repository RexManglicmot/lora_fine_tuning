# app/prepare_data.py
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import cfg

def _rec(t, lab):
    # Build a single SFT-style record used for supervised fine-tuning.
    #   - t : str, is the the input text (already normalized).
    #   - lab : str, is the the gold label for this text.

    # Returns a dictionary
    # A conversation-shaped dict with "system", "user", and "assistant" keys:
    #   - "system": instruction to act as a medical text classifier.
    #   - "user": includes the raw text and the set of allowed labels.
    #   - "assistant": the single correct label (ground truth).

    # Takes the iterable cfg.labels (e.g., ["Lung_Cancer", "Colon_Cancer"]) and makes one string by joining items with ", " between them.
    labs = ", ".join(cfg.labels)
    return {
        "system": "You are a medical text classifier. Choose exactly one label.",
        "user": f"Text: {t}\nLabels: [{labs}]\nReply with one label only.",
        "assistant": lab,
    }

# Write a list of dict rows to a JSONL file (UTF-8), one JSON object per line.
# This is the common on-disk format for SFT datasets.
def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Append per-class counts for a given split to a rolling CSV.
# Allows to verify stratification and class balance across splits.
def _append_class_counts(split_name, y, out_csv):
    
    # Build a 2-column DataFrame (split, label) → count how often each (split,label) pair appears,
    # then convert the counts back into a regular DataFrame with a 'count' column.
    df = pd.DataFrame({"split": split_name, "label": y}).value_counts().reset_index(name="count")
    
    # Write header only on first creation of the CSV (True if file doesn't exist yet; False when appending).
    header = not out_csv.exists()
    df.to_csv(out_csv, mode="a", header=header, index=False)

def main():
    # Loads raq data
    df = pd.read_csv(cfg.paths.raw_csv)
    # Create error if missing
    if "text" not in df or "label" not in df:
        raise ValueError("CSV must contain columns: 'text' and 'label'.")

    # Basic normalization + filtering to the configured label set 
    # Normalize whitespace in text and strip leading/trailing spaces.
    df["text"]  = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    
    # Normalize label strings to be safe (strip only; mapping assumed done upstream or via cfg.labels)
    df["label"] = df["label"].astype(str).str.strip()
    
    # Drop rows with empty text and keep only labels that are in cfg.labels
    df = df[(df["text"] != "") & (df["label"].isin(cfg.labels))].copy()

    # Exact dedupe on normalized text
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Stratify 80/10/10 split to preserve class balance across splits
    X, y = df["text"], df["label"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=cfg.seed, stratify=y)
    X_va, X_te,  y_va, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=cfg.seed, stratify=y_tmp)

    # Write SFT JSONL artifacts (train/val/test) 
    # These are ready to be consumed by training scripts (e.g., LoRA/SFT pipelines).
    out = Path(cfg.paths.splits_dir); out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", [_rec(t, l) for t, l in zip(X_tr, y_tr)])
    _write_jsonl(out / "val.jsonl",   [_rec(t, l) for t, l in zip(X_va, y_va)])
    _write_jsonl(out / "test.jsonl",  [_rec(t, l) for t, l in zip(X_te, y_te)])

    # Build a tiny few-shot pool (k per class) for prompting/debugging
    # This helps create consistent few-shot prompts separate from train/val/test files.
    k = 4
    seen = {lab: 0 for lab in cfg.labels}; few = []
    for t, l in zip(X_tr, y_tr):
        if seen[l] < k:
            few.append({"text": t, "label": l}); seen[l] += 1
    (out / "fewshot.json").write_text(json.dumps(few, ensure_ascii=False, indent=2), encoding="utf-8")

    # Minimal metrics artifacts (class balance per split)
    # These are convenient for reports/plots later without re-reading large files.
    mdir = Path(cfg.paths.metrics_dir); mdir.mkdir(parents=True, exist_ok=True)
    _append_class_counts("train", y_tr, mdir / "class_counts.csv")
    _append_class_counts("val",   y_va, mdir / "class_counts.csv")
    _append_class_counts("test",  y_te, mdir / "class_counts.csv")

    # Simple cross-split leakage check using a stable text hash ---
    # If any of these counts are > 0, the same normalized text appears in multiple splits.
    def _h(s):  # tiny stable hash of normalized text
        return hash(" ".join(str(s).split()).lower())

    s_tr = { _h(t) for t in X_tr }  # hashes for train texts
    s_va = { _h(t) for t in X_va }  # hashes for val texts
    s_te = { _h(t) for t in X_te }  # hashes for test texts

    leaks = {
        "train_val":  len(s_tr & s_va), # overlap between train and val
        "train_test": len(s_tr & s_te), # overlap between train and test
        "val_test":   len(s_va & s_te), # overlap between train and test
    }

    # Save leakage summary so it can be surfaced in reports/dashboards.
    pd.DataFrame([leaks]).to_csv(mdir / "leakage_checks.csv", index=False)

    # Print console summary
    print(f"[prepare_data] Done. Train={len(X_tr)} Val={len(X_va)} Test={len(X_te)}. Labels={cfg.labels}")

if __name__ == "__main__":
    main()

# Run python3 -m app.prepare_data
# Files saved in data/processed (the json and jsonl files)
# Was quick
# Next, is utils.py