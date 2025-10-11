# app/plot_results.py
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from app.config import cfg

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def plot_macro_f1_by_method(results_csv: Path, out_png: Path):
    df = pd.read_csv(results_csv)
    methods = df["method"].tolist()
    scores = df["macro_f1"].tolist()

    plt.figure(figsize=(6,4))
    x = np.arange(len(methods))
    plt.bar(x, scores)
    plt.xticks(x, methods, rotation=0)
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1.0)
    plt.title("Macro-F1 by Method")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_confusion(conf_csv: Path, labels: list[str], out_png: Path):
    cm = pd.read_csv(conf_csv, index_col=0).values
    plt.figure(figsize=(5,4.5))
    im = plt.imshow(cm, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.title("LoRA Confusion Matrix (row-normalized)")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i,j]*100:.0f}%", ha="center", va="center", fontsize=8, color="black")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_macro_f1_and_accuracy(results_csv: Path, out_png: Path):
    df = pd.read_csv(results_csv)  # expects columns: method, macro_f1, accuracy
    methods = df["method"].tolist()
    y_f1 = df["macro_f1"].to_numpy(dtype=float)
    y_acc = df["accuracy"].to_numpy(dtype=float)

    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - width/2, y_f1, width, label="Macro-F1")
    b2 = ax.bar(x + width/2, y_acc, width, label="Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=0)
    ax.set_ylabel("Score")
    ax.set_title("Macro-F1 and Accuracy by Method")
    ax.set_ylim(0, 1.05)                 # headroom so labels aren’t clipped
    ax.margins(y=0.02)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
    #ax.grid(axis="y", linewidth=0.3)

    # value labels on bars
    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9)
    _annotate(b1); _annotate(b2)

    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def plot_val_curve_if_exists(metrics_dir: Path, out_png: Path):
    best_meta_p = metrics_dir / "lora_best.json"
    if not best_meta_p.exists():
        return
    r = _read_json(best_meta_p)["best_r"]
    train_p = metrics_dir / f"lora_r{r}_train.json"
    if not train_p.exists():
        return

    curve = _read_json(train_p).get("val_macro_f1_curve", [])
    if not curve:
        return

    xs = np.arange(1, len(curve) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, curve, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Macro-F1")
    ax.set_title(f"Validation Macro-F1 per Epoch (LoRA r={r})")
    ax.grid(True, linewidth=0.3)

    # --- headroom so the last point isn't clipped ---
    top = max(1.02, float(np.max(curve)) * 1.02)  # 2% headroom; at least 1.02
    ax.set_ylim(0, top)
    ax.margins(y=0.02)                             # tiny padding
    # (nice-to-have) integer epoch ticks + side margins
    ax.set_xlim(0.5, len(curve) + 0.5)
    ax.set_xticks(xs)
    # ------------------------------------------------

    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def main():
    metrics_dir = Path(cfg.paths.metrics_dir)
    figs_dir    = Path(cfg.paths.figures_dir)  # or cfg.paths.plots_dir if that's what you use
    _ensure_dir(metrics_dir)
    _ensure_dir(figs_dir)

    # Required plots
    plot_confusion(metrics_dir / "confusion_lora.csv", cfg.labels, figs_dir / "confusion_matrix_lora.png")

    # Grouped bar: Macro-F1 + Accuracy on ONE chart
    plot_macro_f1_and_accuracy(
        results_csv = metrics_dir / "results.csv",                     # <-- FIXED
        out_png     = figs_dir / "method_bars_macroF1_accuracy.png"    # <-- FIXED
    )

    # Optional plots
    plot_val_curve_if_exists(metrics_dir, figs_dir / "val_macro_f1_per_epoch.png")


    print("[plot_results] Saved figures to", figs_dir.resolve())

if __name__ == "__main__":
    main()

