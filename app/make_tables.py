# app/make_tables.py
import json, ast
from pathlib import Path
import pandas as pd
from app.config import cfg

# ---------- formatting helpers ----------
def _fmt(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"

def _fmt1(x):  # one decimal
    return _fmt(x, nd=1)

def _fmt2(x):  # two decimals
    return _fmt(x, nd=2)

def _fmt_p(p, B=None):
    try:
        p = float(p)
        if p == 0.0 and B:
            return f"< {1.0/float(B):.4f}"
        return f"{p:.3g}" if p < 0.001 else f"{p:.4f}"
    except Exception:
        return "—"

def _parse_latency(s):
    """Parse 'p50,p95' -> ('p50','p95') strings; be forgiving."""
    if not isinstance(s, str):
        return ("—", "—")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return ("—", "—")
    return (_fmt(parts[0], nd=1), _fmt(parts[1], nd=1))

def _parse_per_class_f1(cell):
    """cell is a JSON string (sometimes with double-escaped quotes)."""
    if isinstance(cell, dict):
        return {k: float(v) for k, v in cell.items()}
    if not isinstance(cell, str):
        return {}
    try:
        d = json.loads(cell)
    except Exception:
        try:
            d = ast.literal_eval(cell)
        except Exception:
            return {}
    try:
        return {k: float(v) for k, v in d.items()}
    except Exception:
        return {}

def _short_label(lab: str) -> str:
    """Nice header label: 'Colon_Cancer' -> 'Colon'."""
    if "_" in lab:
        return lab.split("_", 1)[0].capitalize()
    return lab.replace("_", " ").strip().title()

# ---------- table builders ----------
def build_results_table_md(results_csv: Path) -> str:
    if not results_csv.exists():
        raise FileNotFoundError(f"Missing {results_csv} (run app.evaluate first).")

    df = pd.read_csv(results_csv)

    need = {"method","macro_f1","accuracy","per_class_f1",
            "latency_p50_p95_ms","tokens_per_sec","trainable_params_mb","training_time_wall"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{results_csv} missing columns: {missing}")

    # Determine class order from cfg.labels if available; else infer from first row
    labels = list(getattr(cfg, "labels", [])) or list(_parse_per_class_f1(df["per_class_f1"].iloc[0]).keys())
    short_labels = [_short_label(l) for l in labels]

    # --- Header ---
    hdr = ["Method","Macro-F1","Accuracy"]
    for sl in short_labels:
        hdr.append(f"F1 ({sl})")
    hdr += ["Latency p50 / p95 (ms)","Tokens/s","Trainable Params (MB)","Train Time (min)"]

    out = []
    out.append("| " + " | ".join(hdr) + " |")
    out.append("|" + "|".join(["---"] + [":---:"]*(len(hdr)-1)) + "|")

    for _, row in df.iterrows():
        per_f1 = _parse_per_class_f1(row["per_class_f1"])
        f1s = []
        for lab in labels:
            f1s.append(_fmt(per_f1.get(lab, None), nd=3))
        p50, p95 = _parse_latency(str(row["latency_p50_p95_ms"]))
        tokens = _fmt2(row["tokens_per_sec"])
        mb = _fmt1(row["trainable_params_mb"])
        mins = _fmt1(float(row["training_time_wall"]) / 60.0)

        cells = [
            str(row["method"]),
            _fmt(row["macro_f1"], 3),
            _fmt(row["accuracy"], 3),
            *f1s,
            f"{p50} / {p95}",
            tokens,
            mb,
            mins
        ]
        out.append("| " + " | ".join(cells) + " |")

    return "\n".join(out)

def build_stats_table_md(stats_json: Path) -> str:
    if not stats_json.exists():
        # Be graceful: return an empty section with a hint
        return "_(Run evaluation with stats enabled to generate `stat_tests.json`.)_"

    stats = json.loads(stats_json.read_text(encoding="utf-8"))

    mc = stats.get("mcnemar", {})
    b = mc.get("b"); c = mc.get("c")
    delta_acc = mc.get("delta_accuracy")
    p_mcnemar = mc.get("p", mc.get("p_exact"))

    bs = stats.get("bootstrap_macro_f1", {})
    d_macro = bs.get("delta_macro_f1")
    ci = bs.get("ci95", [None, None])
    p_boot = bs.get("p")
    B = bs.get("B")

    out = []
    out.append("| Test | Metric | Effect (LoRA − Base) | 95% CI | p-value | Notes |")
    out.append("|---|---|---:|---|---:|---|")

    # Bootstrap row
    ci_txt = "—"
    if ci and ci[0] is not None and ci[1] is not None:
        ci_txt = f"[{_fmt(ci[0], 4)}, {_fmt(ci[1], 4)}]"
    out.append(
        f"| Paired bootstrap | Macro-F1 "
        f"| **+{_fmt(d_macro, 4)}** "
        f"| **{ci_txt}** "
        f"| **{_fmt_p(p_boot, B)}** "
        f"| {f'{int(B)} resamples' if B else ''} |"
    )

    # McNemar row
    notes = f"b={b}, c={c}" if (b is not None and c is not None) else ""
    out.append(
        f"| McNemar (paired) | Accuracy "
        f"| **+{_fmt(delta_acc, 4)}** "
        f"| — "
        f"| **{_fmt_p(p_mcnemar)}** "
        f"| {notes} |"
    )

    return "\n".join(out)

def main():
    metrics_dir = Path(cfg.paths.metrics_dir)
    results_csv = metrics_dir / "results.csv"
    stats_json  = metrics_dir / "stat_tests.json"

    results_md = build_results_table_md(results_csv)
    stats_md   = build_stats_table_md(stats_json)

    (metrics_dir / "results_table.md").write_text("### Results\n" + results_md + "\n", encoding="utf-8")
    (metrics_dir / "stats_table.md").write_text("### Statistical Tests\n" + stats_md + "\n", encoding="utf-8")
    (metrics_dir / "tables.md").write_text("### Results\n" + results_md + "\n\n### Statistical Tests\n" + stats_md + "\n", encoding="utf-8")

    print("\n### Results\n")
    print(results_md)
    print("\n### Statistical Tests\n")
    print(stats_md)
    print(f"\n[wrote] {metrics_dir / 'results_table.md'}")
    print(f"[wrote] {metrics_dir / 'stats_table.md'}")
    print(f"[wrote] {metrics_dir / 'tables.md'}")

if __name__ == "__main__":
    main()
