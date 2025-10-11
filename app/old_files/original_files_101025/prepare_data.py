# app/prepare_data.py
import json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from app.config import cfg

def _rec(t, lab):
    labs = ", ".join(cfg.labels)
    return {
        "system": "You are a medical text classifier. Choose exactly one label.",
        "user": f"Text: {t}\nLabels: [{labs}]\nReply with one label only.",
        "assistant": lab,
    }

def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _append_class_counts(split_name, y, out_csv):
    df = pd.DataFrame({"split": split_name, "label": y}).value_counts().reset_index(name="count")
    header = not out_csv.exists()
    df.to_csv(out_csv, mode="a", header=header, index=False)

def main():
    df = pd.read_csv(cfg.paths.raw_csv)
    if "text" not in df or "label" not in df:
        raise ValueError("CSV must contain columns: 'text' and 'label'.")

    # normalize + filter
    df["text"]  = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"].isin(cfg.labels))].copy()

    # exact dedupe on normalized text
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # stratified 80/10/10
    X, y = df["text"], df["label"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=cfg.seed, stratify=y)
    X_va, X_te,  y_va, y_te  = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=cfg.seed, stratify=y_tmp)

    # write SFT jsonl (ready for LoRA) + fewshot
    out = Path(cfg.paths.splits_dir); out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", [_rec(t, l) for t, l in zip(X_tr, y_tr)])
    _write_jsonl(out / "val.jsonl",   [_rec(t, l) for t, l in zip(X_va, y_va)])
    _write_jsonl(out / "test.jsonl",  [_rec(t, l) for t, l in zip(X_te, y_te)])

    k = 4
    seen = {lab: 0 for lab in cfg.labels}; few = []
    for t, l in zip(X_tr, y_tr):
        if seen[l] < k:
            few.append({"text": t, "label": l}); seen[l] += 1
    (out / "fewshot.json").write_text(json.dumps(few, ensure_ascii=False, indent=2), encoding="utf-8")

    # minimal artifacts to support planned metrics later
    mdir = Path(cfg.paths.metrics_dir); mdir.mkdir(parents=True, exist_ok=True)
    _append_class_counts("train", y_tr, mdir / "class_counts.csv")
    _append_class_counts("val",   y_va, mdir / "class_counts.csv")
    _append_class_counts("test",  y_te, mdir / "class_counts.csv")

    # exact-hash leakage check across splits (should be 0)
    def _h(s):  # tiny stable hash of normalized text
        return hash(" ".join(str(s).split()).lower())

    s_tr = { _h(t) for t in X_tr }
    s_va = { _h(t) for t in X_va }
    s_te = { _h(t) for t in X_te }

    leaks = {
        "train_val":  len(s_tr & s_va),
        "train_test": len(s_tr & s_te),
        "val_test":   len(s_va & s_te),
    }


    pd.DataFrame([leaks]).to_csv(mdir / "leakage_checks.csv", index=False)

    print(f"[prepare_data] Done. Train={len(X_tr)} Val={len(X_va)} Test={len(X_te)}. Labels={cfg.labels}")

if __name__ == "__main__":
    main()




# # app/prepare_data.py
# from __future__ import annotations
# import json, os, hashlib
# from pathlib import Path
# from collections import defaultdict

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from app.config import cfg  # must expose `cfg` loaded from repo-root config.yaml


# # ---------- helpers ----------
# def _ensure_dirs():
#     Path(cfg.paths.splits_dir).mkdir(parents=True, exist_ok=True)
#     Path(cfg.paths.metrics_dir).mkdir(parents=True, exist_ok=True)
#     Path(cfg.paths.outputs_dir).mkdir(parents=True, exist_ok=True)

# def _canon_ws(s: str) -> str:
#     return " ".join(str(s).split()).strip()

# def _detect_text_col(df: pd.DataFrame) -> str:
#     obj = df.select_dtypes(include=["object"]).columns.tolist()
#     if not obj: raise ValueError("No object/text columns found.")
#     # pick the column with largest average length
#     return max(obj, key=lambda c: df[c].astype(str).str.len().mean())

# def _normalize_label(lbl: str, allowed: list[str]) -> str | None:
#     if pd.isna(lbl): return None
#     s = _canon_ws(str(lbl))
#     # try exact first
#     if s in allowed: return s
#     # simple case-insensitive match
#     lower_map = {x.lower(): x for x in allowed}
#     return lower_map.get(s.lower(), None)

# def _make_record(text: str, label: str) -> dict:
#     labels_join = ", ".join(cfg.labels)
#     return {
#         "system": "You are a medical text classifier. Choose exactly one label.",
#         "user": f"Text: {text}\nLabels: [{labels_join}]\nReply with one label only.",
#         "assistant": label,
#     }

# def _hash_text(s: str) -> str:
#     return hashlib.sha1(_canon_ws(s).lower().encode("utf-8")).hexdigest()


# # ---------- dedupe (exact + near) ----------
# def _dedupe(df: pd.DataFrame, text_col: str, sim_thr: float = 0.95) -> tuple[pd.DataFrame, dict]:
#     stats = {"n_before": len(df), "exact_removed": 0, "near_removed": 0}

#     # exact dedupe
#     df["_norm"] = df[text_col].astype(str).map(_canon_ws).str.lower()
#     before = len(df)
#     df = df.drop_duplicates(subset=["_norm"])
#     stats["exact_removed"] = before - len(df)

#     # block by first 64 chars to keep it cheap
#     df["_block"] = df["_norm"].str.slice(0, 64)
#     to_drop = set()

#     for _, g in df.groupby("_block"):
#         if len(g) < 2: continue
#         # small groups only -> TF-IDF inside block
#         texts = g["_norm"].tolist()
#         vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
#         X = vec.fit_transform(texts)
#         sim = cosine_similarity(X)
#         # mark duplicates beyond the first keep
#         idxs = g.index.tolist()
#         kept = set()
#         for i in range(len(texts)):
#             if idxs[i] in to_drop: continue
#             if idxs[i] in kept: continue
#             kept.add(idxs[i])
#             # any j > i with high sim → drop
#             for j in range(i + 1, len(texts)):
#                 if sim[i, j] >= sim_thr:
#                     to_drop.add(idxs[j])

#     if to_drop:
#         stats["near_removed"] = len(to_drop)
#         df = df.drop(index=list(to_drop))

#     df = df.drop(columns=["_norm", "_block"], errors="ignore").reset_index(drop=True)
#     stats["n_after"] = len(df)
#     return df, stats


# # ---------- write helpers ----------
# def _write_jsonl(path: Path, rows: list[dict]):
#     with path.open("w", encoding="utf-8") as f:
#         for r in rows:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")

# def _save_class_counts(split_name: str, labels: list[str], y: list[str], out_csv: Path):
#     df = pd.DataFrame({"split": split_name, "label": y}).value_counts().reset_index(name="count")
#     df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)

# def _save_dedupe_stats(stats: dict, out_json: Path):
#     with out_json.open("w", encoding="utf-8") as f:
#         json.dump(stats, f, indent=2)


# # ---------- main ----------
# def main():
#     _ensure_dirs()
#     raw_csv = Path(cfg.paths.raw_csv)
#     if not raw_csv.exists():
#         raise FileNotFoundError(f"Missing raw CSV at {raw_csv}")

#     # load
#     df = pd.read_csv(raw_csv)
#     # detect columns
#     text_col = _detect_text_col(df)
#     # assume there is a label column named like any of cfg.labels present; otherwise try 'label'
#     if "label" in df.columns:
#         label_col = "label"
#     else:
#         # try to find a column that matches labels directly
#         cand = [c for c in df.columns if set(df[c].astype(str).unique()) & set(cfg.labels)]
#         label_col = cand[0] if cand else "label"
#         if label_col not in df.columns:
#             raise ValueError("No label column found. Add a 'label' column with target classes.")

#     # normalize text & labels
#     df[text_col] = df[text_col].astype(str).map(_canon_ws)
#     df[label_col] = df[label_col].apply(lambda x: _normalize_label(x, cfg.labels))

#     # drop bad rows
#     df = df[df[text_col].str.len() > 0].dropna(subset=[label_col])
#     df = df[df[label_col].isin(cfg.labels)].reset_index(drop=True)

#     # dedupe
#     df, dstats = _dedupe(df, text_col=text_col, sim_thr=0.95)

#     # stratified 80/10/10
#     X = df[text_col].tolist()
#     y = df[label_col].tolist()
#     X_train, X_tmp, y_train, y_tmp = train_test_split(
#         X, y, test_size=0.2, random_state=cfg.seed, stratify=y
#     )
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_tmp, y_tmp, test_size=0.5, random_state=cfg.seed, stratify=y_tmp
#     )

#     # build SFT records
#     train_recs = [_make_record(t, l) for t, l in zip(X_train, y_train)]
#     val_recs   = [_make_record(t, l) for t, l in zip(X_val,   y_val  )]
#     test_recs  = [_make_record(t, l) for t, l in zip(X_test,  y_test )]

#     # write splits
#     splits_dir = Path(cfg.paths.splits_dir)
#     _write_jsonl(splits_dir / "train.jsonl", train_recs)
#     _write_jsonl(splits_dir / "val.jsonl",   val_recs)
#     _write_jsonl(splits_dir / "test.jsonl",  test_recs)

#     # few-shot exemplars (k=4 per class from train)
#     k = 4
#     by_class = defaultdict(list)
#     for t, l in zip(X_train, y_train):
#         if len(by_class[l]) < k:
#             by_class[l].append({"text": t, "label": l})
#     few = []
#     # keep label order consistent with cfg.labels
#     for lab in cfg.labels:
#         few.extend(by_class.get(lab, []))
#     with (splits_dir / "fewshot.json").open("w", encoding="utf-8") as f:
#         json.dump(few, f, ensure_ascii=False, indent=2)

#     # metrics artifacts
#     metrics_dir = Path(cfg.paths.metrics_dir)
#     _save_class_counts("train", cfg.labels, y_train, metrics_dir / "class_counts.csv")
#     _save_class_counts("val",   cfg.labels, y_val,   metrics_dir / "class_counts.csv")
#     _save_class_counts("test",  cfg.labels, y_test,  metrics_dir / "class_counts.csv")

#     # dedupe stats
#     _save_dedupe_stats(dstats, metrics_dir / "dedupe_stats.json")

#     # simple leakage check (exact text hash overlap across splits should be 0)
#     def _hash_set(texts): return { _hash_text(t) for t in texts }
#     leaks = {
#         "train∩val":  len(_hash_set(X_train) & _hash_set(X_val)),
#         "train∩test": len(_hash_set(X_train) & _hash_set(X_test)),
#         "val∩test":   len(_hash_set(X_val)   & _hash_set(X_test)),
#     }
#     pd.DataFrame([leaks]).to_csv(metrics_dir / "leakage_checks.csv", index=False)

#     print("[prepare_data] Done.",
#           f"Train={len(X_train)} Val={len(X_val)} Test={len(X_test)}.",
#           f"Dropped exact={dstats['exact_removed']} near={dstats['near_removed']}.",
#           f"Leaks (exact-hash): {leaks}", sep="\n")


# if __name__ == "__main__":
#     main()
