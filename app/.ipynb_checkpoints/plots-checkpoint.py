# app/plot.py
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from app.config import cfg
from matplotlib.colors import LinearSegmentedColormap, to_rgb

def _single_hue_cmap(hex_color: str, name: str = "single_hue"):
    rgb = to_rgb(hex_color)
    return LinearSegmentedColormap.from_list(name, [(1,1,1), rgb], N=256)

# ---------- minimal, consistent theme ----------
PALETTE = {
    "macro_f1": "#3BA79E",  # teal
    "accuracy": "#D7EBE9",  # lime-yellow
    "text_dark": "#1F1F1F",
    "cm_cmap": "Greys",
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C7C7C7")
    ax.spines["bottom"].set_color("#C7C7C7")
    ax.tick_params(axis="both", length=0)
    ax.margins(y=0.02)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def _savefig(fig: plt.Figure, path: Path):
    """Create parent dir, save, and print the absolute path so you can find it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"[save] {path.resolve()}")


# ---------- plots ----------

def plot_confusion(conf_csv: Path, labels: list[str], out_png: Path):
    if not conf_csv.exists():
        print(f"[warn] Missing {conf_csv}; skipping confusion matrix.")
        return

    cm = pd.read_csv(conf_csv, index_col=0).values
    n = len(labels)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    # modern colormap API (fixes deprecation)
    # cmap = mpl.colormaps[PALETTE["cm_cmap"]]
    cmap = _single_hue_cmap(PALETTE["macro_f1"])
    im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # draw black edges around each cell
    for i in range(n):
        for j in range(n):
            ax.add_patch(plt.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=0.5,
                joinstyle="miter", zorder=3
            ))
    # Outer frame black too
    # for s in ax.spines.values():
    #     s.set_visible(True)
    #     s.set_color("black")
    #     s.set_linewidth(1.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized", rotation=90, va="center", color=PALETTE["text_dark"])
    cbar.ax.tick_params(colors=PALETTE["text_dark"])

    ax.set_xticks(np.arange(n)); ax.set_xticklabels(labels, rotation=30, ha="right", color=PALETTE["text_dark"])
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(labels, color=PALETTE["text_dark"])
    ax.set_title("LoRA Confusion Matrix (row-normalized)", color=PALETTE["text_dark"])

    # Auto-contrast annotations
    for i in range(n):
        for j in range(n):
            v = float(cm[i, j])
            r, g, b, _ = cmap(v)
            lum = 0.2126*r + 0.7152*g + 0.0722*b
            text_color = "#FFFFFF" if lum < 0.45 else "#000000"
            ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=10, color=text_color)

    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    _savefig(fig, out_png)

def plot_macro_f1_and_accuracy(results_csv: Path, out_png: Path):
    if not results_csv.exists():
        print(f"[warn] Missing {results_csv}; skipping Macro-F1 & Accuracy bars.")
        return

    df = pd.read_csv(results_csv)
    methods = df["method"].tolist()
    y_f1 = df["macro_f1"].to_numpy(dtype=float)
    y_acc = df["accuracy"].to_numpy(dtype=float)

    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    b1 = ax.bar(x - width/2, y_f1, width, label="Macro-F1", color=PALETTE["macro_f1"], edgecolor="black")
    b2 = ax.bar(x + width/2, y_acc, width, label="Accuracy", color=PALETTE["accuracy"], edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Score")
    ax.set_title("Macro-F1 and Accuracy by Method")
    ax.set_ylim(0, max(1.02, float(np.max([y_f1.max(), y_acc.max()])) * 1.05))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    def _annotate(bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.015, f"{h:.3f}",
                    ha="center", va="bottom")
    _annotate(b1); _annotate(b2)

    _polish_axes(ax)
    _savefig(fig, out_png)

def plot_val_curve_if_exists(metrics_dir: Path, out_png: Path):
    best_meta_p = metrics_dir / "lora_best.json"
    if not best_meta_p.exists():
        print(f"[warn] Missing {best_meta_p}; skipping val curve.")
        return
    r = _read_json(best_meta_p)["best_r"]
    train_p = metrics_dir / f"lora_r{r}_train.json"
    if not train_p.exists():
        print(f"[warn] Missing {train_p}; skipping val curve.")
        return

    curve = _read_json(train_p).get("val_macro_f1_curve", [])
    if not curve:
        print("[warn] Empty val_macro_f1_curve; skipping val curve.")
        return

    xs = np.arange(1, len(curve) + 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        xs, curve, marker="o", linewidth=1.6, color=PALETTE["macro_f1"],
        markeredgecolor="black", markerfacecolor="#D7EBE9"
    )

    # --- add value labels above each dot ---
    for x, y in zip(xs, curve):
        ax.annotate(f"{y:.3f}",
                xy=(x, y),
                xytext=(0, -15),               # ↓ 10 px below the point
                textcoords="offset points",
                ha="center", va="top",
                fontsize=10)

    # give room under the lowest point so labels aren't clipped
    bottom = max(0.0, float(np.min(curve)) - 0.12)   # ~0.12 headroom
    top    = max(1.05, float(np.max(curve)) * 1.05)
    ax.set_ylim(bottom, top)
    # ---------------------------------------

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Macro-F1")
    ax.set_title(f"Validation Macro-F1 per Epoch (LoRA r={r})")

    # give extra headroom so labels never get clipped
    top = max(1.05, float(np.max(curve)) * 1.05)
    ax.set_ylim(0, top)
    ax.set_xlim(0.5, len(curve) + 0.5)
    ax.set_xticks(xs)

    _polish_axes(ax)
    _savefig(fig, out_png)

def plot_efficiency_panel(results_csv: Path, out_png: Path):
    """Tiny panel: Trainable params (MB) and Training time (min) by method."""
    if not results_csv.exists():
        print(f"[warn] Missing {results_csv}; skipping efficiency panel.")
        return

    df = pd.read_csv(results_csv)
    required = {"method", "trainable_params_mb", "training_time_wall"}
    if not required.issubset(df.columns):
        print(f"[warn] {results_csv} missing {required - set(df.columns)}; skipping.")
        return

    methods = df["method"].tolist()
    mb = df["trainable_params_mb"].to_numpy(dtype=float)
    # `training_time_wall` was written in **seconds** by train_lora.py → convert to minutes
    mins = (df["training_time_wall"].to_numpy(dtype=float) / 60.0)

    x = np.arange(len(methods))
    w = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    ax0, ax1 = axes

    # --- Trainable Params (MB) ---
    b0 = ax0.bar(x, mb, width=w, color=PALETTE["macro_f1"], edgecolor="black")
    ax0.set_xticks(x); ax0.set_xticklabels(methods)
    ax0.set_ylabel("MB")
    ax0.set_title("Trainable Parameters (MB)")
    ax0.set_ylim(0, max(1.02, float(mb.max()) * 1.15))
    for bar in b0:
        h = bar.get_height()
        ax0.text(bar.get_x() + bar.get_width()/2, h + (0.03 * max(1.0, mb.max())),
                 f"{h:.1f}", ha="center", va="bottom", fontsize=10)
    _polish_axes(ax0)

    # --- Training Time (min) ---
    b1 = ax1.bar(x, mins, width=w, color=PALETTE["accuracy"], edgecolor="black")
    ax1.set_xticks(x); ax1.set_xticklabels(methods)
    ax1.set_ylabel("Minutes")
    ax1.set_title("Training Time to Best (min)")
    ax1.set_ylim(0, max(1.02, float(mins.max()) * 1.15) if mins.size and mins.max() > 0 else 1.0)
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + (0.03 * max(1.0, mins.max())),
                 f"{h:.1f}", ha="center", va="bottom", fontsize=10)
    _polish_axes(ax1)

    fig.tight_layout()
    _savefig(fig, out_png)


# ---------- main ----------
def main():
    _apply_theme()

    metrics_dir = Path(cfg.paths.metrics_dir)
    figs_dir    = Path(cfg.paths.figures_dir)
    _ensure_dir(metrics_dir)
    _ensure_dir(figs_dir)

    # Required plots
    plot_confusion(metrics_dir / "confusion_lora.csv", cfg.labels, figs_dir / "confusion_matrix_lora.png")

    # Grouped bar: Macro-F1 + Accuracy on ONE chart
    plot_macro_f1_and_accuracy(
        results_csv = metrics_dir / "results.csv",
        out_png     = figs_dir / "method_bars_macroF1_accuracy.png"
    )

    # Val curve
    plot_val_curve_if_exists(metrics_dir, figs_dir / "val_macro_f1_per_epoch.png")
    
    # Efficient bars
    plot_efficiency_panel(metrics_dir / "results.csv", figs_dir / "efficiency_panel.png")

    print("[plot_results] Saved figures to", figs_dir.resolve())

if __name__ == "__main__":
    main()

# Run python3 -m app.plots
# Next, is make_tables.py