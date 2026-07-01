"""
WeightDistance.py — figures from the weight-distance summary CSVs. No weight-loading
or cosine-computation happens here — pure read-csv / render-matplotlib.

For each pretraining arm we compare the fine-tuned A model's transformer body to
its pretrained init, against the random-init B baseline. The meaningful quantity
is the RETENTION MARGIN = cos(A,pre) - cos(B,pre): how much MORE A stays aligned
to its init than an unrelated model does.
Produces: global retention bars, per-layer retention profiles, and relL2 movement.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUTDIR = "Plots/WeightDistance"
INDIR = "Results/WeightDistance"
DPI = 300

# arm -> (summary csv, pretty name, color)
ARMS = [
    ("weight_distance_rule30_summary.csv",    "Rule30 (local)",            "#d1495b"),
    ("weight_distance_carryonly_summary.csv", "Carry (matched)",           "#1b9e77"),
    ("weight_distance_rollout_summary.csv",   "Rollout (mismatched)",      "#6a3d9a"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 120,
})

LAYER_ORDER = ["layer 0","layer 1","layer 2","layer 3","layer 4","layer 5","final_norm"]


def load_summary(fn):
    """Read one weight_distance_*_summary.csv (as written by WeightDistance.py's
    main()) into {row_label: {column: float}}, e.g. d["layer 3"]["cosA_pre_mean"]."""

    out = {}
    for r in csv.DictReader(open(os.path.join(INDIR, fn))):
        lab = r["layer"]
        out[lab] = {}

        for k, v in r.items():
            if k == "layer":
                continue
            out[lab][k] = float(v) if v not in ("", None) else None

    return out


def margin(d, lab):
    """Retention margin cos(A,pre) - cos(B,pre) at one row (layer or GLOBAL),
    with std propagated via sqrt(sa^2 + sb^2) — the comparable, cross-arm quantity."""

    a = d[lab]["cosA_pre_mean"]; b = d[lab]["cosB_pre_mean"]
    sa = d[lab]["cosA_pre_std"] or 0.0; sb = d[lab]["cosB_pre_std"] or 0.0

    return a - b, (sa**2 + sb**2) ** 0.5


def fig_global_retention(data):
    """Two-panel headline figure: raw global cos(A,pre) vs cos(B,pre) per arm
    (left, illustrates each arm relative to its own floor) and the retention
    MARGIN per arm (right, the actual cross-arm-comparable result)."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    names = [n for _, n, _ in ARMS]
    colors = [c for _, _, c in ARMS]

    # left: raw global cos(A,pre) vs cos(B,pre)
    x = range(len(names))
    a_vals = [data[n]["GLOBAL"]["cosA_pre_mean"] for n in names]
    a_err  = [data[n]["GLOBAL"]["cosA_pre_std"]  for n in names]
    b_vals = [data[n]["GLOBAL"]["cosB_pre_mean"] for n in names]
    b_err  = [data[n]["GLOBAL"]["cosB_pre_std"]  for n in names]
    w = 0.36

    ax1.bar([i - w/2 for i in x], a_vals, w, yerr=a_err, capsize=4,
            color=colors, edgecolor="white", label="cos(A, pre)")
    ax1.bar([i + w/2 for i in x], b_vals, w, yerr=b_err, capsize=4,
            color="#bbbbbb", edgecolor="white", label="cos(B, pre) = floor")
    
    ax1.set_xticks(list(x)); ax1.set_xticklabels(names, fontsize=10)
    ax1.set_ylabel("global cosine to pretrained init")
    ax1.set_title("Alignment to init: fine-tuned A vs random B")
    ax1.legend(fontsize=10)

    # right: the retention MARGIN (the comparable quantity)
    m = [margin(data[n], "GLOBAL")[0] for n in names]
    me = [margin(data[n], "GLOBAL")[1] for n in names]
    bars = ax2.bar(x, m, yerr=me, capsize=5, color=colors, edgecolor="white", width=0.6)

    for i, b in enumerate(bars):
        ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f"{m[i]:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax2.set_xticks(list(x)); ax2.set_xticklabels(names, fontsize=10)
    ax2.set_ylabel("retention margin  cos(A,pre) − cos(B,pre)")
    ax2.set_title("Global retention margin\n(higher = more pretraining structure kept)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "wd_fig_global_retention.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_layer_margin(data):
    """Per-layer retention-margin profile, all three arms on one axis — shows
    WHERE (which depth) each arm retains pretraining structure, not just how much globally."""

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    xt = [l for l in LAYER_ORDER]
    xpos = range(len(xt))

    for fn, name, color in ARMS:
        d = data[name]
        ys = []; es = []

        for lab in xt:
            mu, sd = margin(d, lab)
            ys.append(mu); es.append(sd)

        ax.errorbar(xpos, ys, yerr=es, marker="o", lw=2.4, ms=7, capsize=3,
                    color=color, label=name)
        
    ax.axhline(0, color="#666", lw=1)
    ax.set_xticks(list(xpos)); ax.set_xticklabels([x.replace("layer ", "L") for x in xt])
    ax.set_xlabel("transformer body component (depth →)")
    ax.set_ylabel("retention margin  cos(A,pre) − cos(B,pre)")
    ax.set_title("Where pretraining structure is retained, by layer")
    ax.legend(fontsize=10.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "wd_fig_layer_margin.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_rell2(data):
    """Per-layer relative-L2 movement from init, all three arms — a magnitude-
    based complement to the cosine-based retention margin."""

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    xt = [l for l in LAYER_ORDER if l != "final_norm"] + ["final_norm"]
    xpos = range(len(xt))

    for fn, name, color in ARMS:
        d = data[name]
        ys = [d[lab]["relL2_mean"] for lab in xt]
        es = [d[lab]["relL2_std"] or 0.0 for lab in xt]
        ax.errorbar(xpos, ys, yerr=es, marker="s", lw=2.4, ms=6, capsize=3,
                    color=color, label=name)
        
    ax.set_xticks(list(xpos)); ax.set_xticklabels([x.replace("layer ", "L") for x in xt])
    ax.set_xlabel("transformer body component (depth →)")
    ax.set_ylabel("relative L2 distance moved from init")
    ax.set_title("How far weights move from init during fine-tuning\n(higher = more change)")
    ax.legend(fontsize=10.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "wd_fig_rell2.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    """Load all three arms' summary CSVs and generate the three figures. Pure
    plotting entry point — no weight-distance computation happens here."""

    data = {name: load_summary(fn) for fn, name, _ in ARMS}
    fig_global_retention(data)
    fig_layer_margin(data)
    fig_rell2(data)
    print("saved: wd_fig_global_retention.png, wd_fig_layer_margin.png, wd_fig_rell2.png")


if __name__ == "__main__":
    main()