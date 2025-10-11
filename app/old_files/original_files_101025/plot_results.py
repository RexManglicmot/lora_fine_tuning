# app/plot_results.py
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from app.config import cfg

# ---------- minimal, consistent theme ----------
PALETTE = {
    "macro_f1": "#4E79A7",  # blue
        "accuracy": "#F28E2B",  # orange
        "text_dark": "#1F1F1F",
        "cm_cmap": "YlGnBu",
    }

def _apply_theme():
    mpl.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "figure.constrained_layout.use": True,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titlepad": 8,
        "axes.labelpad": 6,
        "axes.edgecolor": "#C7C7C7",
        "text.color": PALETTE["text_dark"],
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
    })

def _polish_axes(ax):
    # remove top/right spines; lighten left/bottom
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C7C7C7")
    ax.spines["bottom"].set_color("#C7C7C7")
    ax.tick_params(axis="both", length=0)  # hide ticks (keep labels)
    ax.margins(y=0.02)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

# ---------- plots ----------
def plot_macro_f1_by_method(results_csv: Path, out_png: Path):
    df = pd.read_csv(results_csv)
    methods = df["method"].tolist()
    scores = df["macro_f1"].to_numpy(dtype=float)

    x = np.arange(len(methods))
    w = 0.55

    fig, ax = plt.subplots(figsize=(7, 4.4))
    bars = ax.bar(x, scores, width=w, color=PALETTE["macro_f1"])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 by Method")
    ax.set_ylim(0, max(1.02, float(scores.max()) * 1.05))

    # value labels
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.015, f"{h:.3f}",
                ha="center", va="bottom")

    _polish_axes(ax)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def plot_confusion(conf_csv: Path, labels: list[str], out_png: Path):
    cm = pd.read_csv(conf_csv, index_col=0).values
    n = len(labels)

    fig, ax = plt.subplots(figsize=(5.6, 5.1))
    im = ax.imshow(cm, cmap=PALETTE["cm_cmap"], vmin=0, vmax=1, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized", rotation=90, va="center")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(labels)
    ax.set_title("LoRA Confusion Matrix (row-normalized)")

    # annotations with automatic contrast
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            txt_color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center", fontsize=9, color=txt_color)

    # tidy frame (square look)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def plot_macro_f1_and_accuracy(results_csv: Path, out_png: Path):
    df = pd.read_csv(results_csv)  # expects columns: method, macro_f1, accuracy
    methods = df["method"].tolist()
    y_f1 = df["macro_f1"].to_numpy(dtype=float)
    y_acc = df["accuracy"].to_numpy(dtype=float)

    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    b1 = ax.bar(x - width/2, y_f1, width, label="Macro-F1", color=PALETTE["macro_f1"])
    b2 = ax.bar(x + width/2, y_acc, width, label="Accuracy", color=PALETTE["accuracy"])

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Macro-F1 and Accuracy by Method")
    ax.set_ylim(0, max(1.02, float(np.max([y_f1.max(), y_acc.max()])) * 1.05))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    # value labels
    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom")
    _annotate(b1); _annotate(b2)

    _polish_axes(ax)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
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
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(xs, curve, marker="o", linewidth=1.6, color=PALETTE["macro_f1"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Macro-F1")
    ax.set_title(f"Validation Macro-F1 per Epoch (LoRA r={r})")

    # breathing room & tidy ticks
    top = max(1.02, float(np.max(curve)) * 1.02)
    ax.set_ylim(0, top)
    ax.set_xlim(0.5, len(curve) + 0.5)
    ax.set_xticks(xs)

    _polish_axes(ax)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

# ---------- main ----------
def main():
    _apply_theme()

    metrics_dir = Path(cfg.paths.metrics_dir)
    figs_dir    = Path(cfg.paths.figures_dir)  # or cfg.paths.plots_dir
    _ensure_dir(metrics_dir)
    _ensure_dir(figs_dir)

    # Required plots
    plot_confusion(metrics_dir / "confusion_lora.csv", cfg.labels, figs_dir / "confusion_matrix_lora.png")

    # Grouped bar: Macro-F1 + Accuracy on ONE chart
    plot_macro_f1_and_accuracy(
        results_csv = metrics_dir / "results.csv",
        out_png     = figs_dir / "method_bars_macroF1_accuracy.png"
    )

    # Optional plots
    plot_val_curve_if_exists(metrics_dir, figs_dir / "val_macro_f1_per_epoch.png")

    print("[plot_results] Saved figures to", figs_dir.resolve())

if __name__ == "__main__":
    main()
