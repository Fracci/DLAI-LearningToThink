"""
plot_positional.py — answer-digit accuracy by position, from the *_positional_accuracy.csv files.

Position 0 = least-significant answer digit (units); higher = more-significant.
Reveals WHERE along the answer each model fails OOD. The expected carry-propagation
signature is a dip in the MIDDLE digits (where carries must chain across the unseen
extra length), with units and top digits staying high. Produces per-length curves
per arm, an arm-overlay at the decisive 6-digit length, and an accuracy heatmap.
ROLLOUT is included but commented out until its CSV exists (see ARMS).
"""
import os
import csv
import statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUTDIR = "Plots/PositionalAccuracy"
INDIR = "Results/PositionalAccuracy"
DPI = 300

# arm -> (csv filename, pretty name, color)
ARMS = [
    ("Rule30_positional_accuracy.csv",     "Rule30 (local)",   "#d1495b"),
    ("carryonly_positional_accuracy.csv",  "Carry (matched)",  "#1b9e77"),
    # ("rollout_positional_accuracy.csv",  "Rollout (mismatched)", "#6a3d9a"),  # ROLLOUT: uncomment when CSV exists
]
OOD = ["5dig", "6dig", "7dig"]   # 'id' is ~100% everywhere; shown only in the per-arm grid

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 12.5, "figure.dpi": 120,
})


def load(fn):
    """Return {eval_set: {pos: (mean, std)}} averaged over seeds (skips nan)."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(os.path.join(INDIR, fn))):
        v = r["accuracy"]
        if v in ("", "nan", "NaN"):
            continue
        try:
            acc[r["eval_set"]][int(r["pos_from_LSB"])].append(float(v))
        except ValueError:
            continue
    out = {}
    for es, posmap in acc.items():
        out[es] = {p: (st.mean(vs), st.pstdev(vs) if len(vs) > 1 else 0.0)
                   for p, vs in posmap.items()}
    return out


def series(es_map, es):
    """Sorted (positions, means, stds) for one eval set."""
    if es not in es_map:
        return [], [], []
    ps = sorted(es_map[es])
    means = [es_map[es][p][0] for p in ps]
    stds = [es_map[es][p][1] for p in ps]
    return ps, means, stds


def fig_per_arm_grid(data):
    """One panel per arm: accuracy vs position, one line per OOD length."""
    arms = list(data.keys())
    fig, axes = plt.subplots(1, len(arms), figsize=(7.0 * len(arms), 5.6), squeeze=False)
    length_colors = {"id": "#999999", "5dig": "#4c9f70", "6dig": "#e08214", "7dig": "#b2182b"}
    for ax, (name, es_map) in zip(axes[0], data.items()):
        for es in ["id"] + OOD:
            ps, means, stds = series(es_map, es)
            if not ps:
                continue
            ax.plot(ps, means, "-o", color=length_colors[es], lw=2.6, ms=6, label=es)
            lo = [m - s for m, s in zip(means, stds)]; hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(ps, lo, hi, color=length_colors[es], alpha=0.12, linewidth=0)
        ax.set_title(name); ax.set_xlabel("answer position (0 = units →)")
        ax.set_ylabel("digit accuracy (%)"); ax.set_ylim(0, 103)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(title="length", loc="lower left")
    fig.suptitle("Answer-digit accuracy by position — the middle-digit dip is the carry-chain failure",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "pos_fig_per_arm.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_arms_overlay(data, es="6dig"):
    """All arms on one axis at the decisive length, to compare positional profiles."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for (name, es_map), (_, _, color) in zip(data.items(), ARMS):
        ps, means, stds = series(es_map, es)
        if not ps:
            continue
        ax.plot(ps, means, "-o", color=color, lw=3.0, ms=7, label=name)
        lo = [m - s for m, s in zip(means, stds)]; hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(ps, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.set_title(f"{es}: answer-digit accuracy by position (mean ± seed std)")
    ax.set_xlabel("answer position (0 = units, → more significant)")
    ax.set_ylabel("digit accuracy (%)"); ax.set_ylim(0, 103)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"pos_fig_overlay_{es}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(data):
    """Heatmap per arm: rows = OOD length, cols = position, color = accuracy."""
    arms = list(data.items())
    fig, axes = plt.subplots(1, len(arms), figsize=(7.2 * len(arms), 4.2), squeeze=False)
    maxpos = max((p for _, em in arms for es in OOD for p in (em.get(es, {}) or {})), default=7) + 1
    for ax, (name, es_map) in zip(axes[0], arms):
        grid = []
        for es in OOD:
            rowvals = [es_map.get(es, {}).get(p, (float("nan"),))[0] for p in range(maxpos)]
            grid.append(rowvals)
        im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
        ax.set_xticks(range(maxpos)); ax.set_xticklabels(range(maxpos))
        ax.set_yticks(range(len(OOD))); ax.set_yticklabels(OOD)
        ax.set_xlabel("answer position (0 = units →)"); ax.set_title(name)
        for i in range(len(OOD)):
            for j in range(maxpos):
                v = grid[i][j]
                if v == v:  # not nan
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=9, color="black")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="digit accuracy (%)")
    fig.suptitle("Where the answer breaks: accuracy heatmap (length × position)",
                 fontsize=16, fontweight="bold", y=1.05)
    fig.savefig(os.path.join(OUTDIR, "pos_fig_heatmap.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    data = {}
    for fn, name, _ in ARMS:
        if os.path.exists(os.path.join(INDIR, fn)):
            data[name] = load(fn)
        else:
            print(f"  [skip] {name}: {fn} not found")
    if not data:
        print("No positional CSVs found."); return
    fig_per_arm_grid(data)
    fig_arms_overlay(data, "6dig")
    fig_arms_overlay(data, "7dig")
    fig_heatmap(data)
    print("saved: pos_fig_per_arm.png, pos_fig_overlay_6dig.png, "
          "pos_fig_overlay_7dig.png, pos_fig_heatmap.png")


if __name__ == "__main__":
    main()