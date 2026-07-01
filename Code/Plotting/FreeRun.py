"""
FreeRun.py — generates figures for the free-running
(non-teacher-forced) evaluation. Reads freerun_results.csv (per-seed rows,
written by FreeRunEval.py) and produces matplotlib PNGs; does no training,
evaluation, or model loading itself.
"""
import os
import csv
import statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "Plots/FreeRun"
INDIR = "Results/FreeRun"
DPI = 300
DETAIL = "freerun_results.csv"

MODELS = ["Rule30", "Rollout", "Carry", "Baseline"]
COLORS = {"Rule30": "#d1495b", "Rollout": "#6a3d9a", "Carry": "#1b9e77", "Baseline": "#8a8a8a"}
DIGITS = [5, 6, 7]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "axes.titlesize": 16, "axes.titleweight": "bold", "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 12, "figure.dpi": 120,
})


def load_detail():
    """Read per-seed rows from freerun_results.csv -> {(model,digits): {metric:[vals]}},
    ready for mean_std() below. Malformed/missing values are silently skipped."""

    d = defaultdict(lambda: defaultdict(list))

    for r in csv.DictReader(open(os.path.join(INDIR, DETAIL))):
        key = (r["model"], int(r["digits"]))
        for m in ("freerun_EM", "freerun_PD", "parseable_pct",
                  "TF_EM_sameckpt", "TF_PD_sameckpt", "drop_EM", "drop_PD"):
            
            try:
                d[key][m].append(float(r[m]))
            except (ValueError, KeyError):
                pass
    return d


def ms(xs):
    """Population mean/std, NaN-safe for empty input."""

    if not xs:
        return (float("nan"), float("nan"))
    
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)


def fig_tf_vs_free(d, metric="PD"):
    """Grouped bars per OOD length: teacher-forced vs free-running all four 
    arms side by side — visualizes the collapse from TF accuracy down to free-running accuracy."""

    fig, axes = plt.subplots(1, len(DIGITS), figsize=(6.0 * len(DIGITS), 5.4), sharey=True)
    tf_key = f"TF_{metric}_sameckpt"; fr_key = f"freerun_{metric}"

    for ax, dig in zip(axes, DIGITS):
        x = range(len(MODELS)); w = 0.38
        tf = [ms(d[(m, dig)][tf_key])[0] for m in MODELS]
        fr = [ms(d[(m, dig)][fr_key])[0] for m in MODELS]
        fre = [ms(d[(m, dig)][fr_key])[1] for m in MODELS]

        ax.bar([i - w/2 for i in x], tf, w, color=[COLORS[m] for m in MODELS],
               alpha=0.45, edgecolor="white", label="teacher-forced")
        ax.bar([i + w/2 for i in x], fr, w, yerr=fre, capsize=3,
               color=[COLORS[m] for m in MODELS], edgecolor="white", label="free-running")
        ax.set_title(f"{dig}-digit"); ax.set_xticks(list(x))
        ax.set_xticklabels(MODELS, rotation=30, ha="right")

        if dig == DIGITS[0]:
            ax.set_ylabel(f"{metric} (%)")
        ax.set_ylim(0, 103)

    # single legend
    from matplotlib.patches import Patch

    handles = [Patch(facecolor="#777", alpha=0.45, label="teacher-forced"),
               Patch(facecolor="#777", label="free-running")]
    axes[-1].legend(handles=handles, loc="upper right")
    fig.suptitle(f"Teacher-forced vs free-running {metric}: the transfer advantage does not survive error accumulation",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"fr_fig_tf_vs_free_{metric}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_drop(d, metric="PD"):
    """Line plot of the error-accumulation drop vs OOD length, one
    line per arm — shows the drop growing with length and being larger 
    for arms with higher TF scores to begin with."""

    fig, ax = plt.subplots(figsize=(10, 6))
    key = f"drop_{metric}"
    x = list(DIGITS)

    for m in MODELS:
        ys = [ms(d[(m, dig)][key])[0] for dig in DIGITS]
        es = [ms(d[(m, dig)][key])[1] for dig in DIGITS]
        ls = "--" if m == "Baseline" else "-"
        ax.errorbar(x, ys, yerr=es, marker="o", lw=2.8, ms=8, capsize=4,
                    color=COLORS[m], ls=ls, label=m)
        
    ax.set_title(f"Error-accumulation drop  (teacher-forced − free-running {metric})")
    ax.set_xlabel("OOD length (digits)"); ax.set_ylabel(f"{metric} drop (pts)")
    ax.set_xticks(DIGITS); ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"fr_fig_drop_{metric}.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_free_pd_and_parseable(d):
    """Two-panel figure: free-run PD by length (left, the signal that survives)
    and parseable_pct by length."""

    fig, (axP, axQ) = plt.subplots(1, 2, figsize=(14, 5.6))
    x = list(DIGITS)

    for m in MODELS:
        pd = [ms(d[(m, dig)]["freerun_PD"])[0] for dig in DIGITS]
        pde = [ms(d[(m, dig)]["freerun_PD"])[1] for dig in DIGITS]
        pr = [ms(d[(m, dig)]["parseable_pct"])[0] for dig in DIGITS]
        ls = "--" if m == "Baseline" else "-"

        axP.errorbar(x, pd, yerr=pde, marker="o", lw=2.8, ms=8, capsize=4,
                     color=COLORS[m], ls=ls, label=m)
        axQ.plot(x, pr, marker="s", lw=2.6, ms=7, color=COLORS[m], ls=ls, label=m)

    axP.set_title("Free-running per-digit accuracy (PD)")
    axP.set_xlabel("OOD length (digits)"); axP.set_ylabel("free-run PD (%)")
    axP.set_xticks(DIGITS); axP.legend(loc="upper right")

    axQ.set_title("Parseable answers (%)\n(low = malformed generations, deflates PD)")
    axQ.set_xlabel("OOD length (digits)"); axQ.set_ylabel("parseable (%)")
    axQ.set_xticks(DIGITS); axQ.set_ylim(0, 103); axQ.legend(loc="lower left")
    
    fig.suptitle("What survives free-running: partial-digit signal, and how often answers are well-formed",
                 fontsize=16, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fr_fig_free_pd_parseable.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    """Load the free-run detail CSV and generate all four figures. Pure
    plotting entry point — no computation of the underlying metrics happens here."""
    
    d = load_detail()
    fig_tf_vs_free(d, "PD")
    fig_tf_vs_free(d, "EM")
    fig_drop(d, "PD")
    fig_free_pd_and_parseable(d)
    print("saved: fr_fig_tf_vs_free_PD.png, fr_fig_tf_vs_free_EM.png, "
          "fr_fig_drop_PD.png, fr_fig_free_pd_parseable.png")


if __name__ == "__main__":
    main()