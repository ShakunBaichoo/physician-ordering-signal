#!/usr/bin/env python
"""Generate Figure 14 — Behavioral Fingerprint from saved results."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).parents[2]
OUT  = ROOT / "1_ordering_paper" / "results" / "O_behavioral"
FIG  = ROOT / "1_ordering_paper" / "results" / "figures"

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "aki":             "AKI",
    "sepsis":          "Sepsis",
}

# Wong (2011) colorblind-safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
SKY    = "#56B4E9"
RED    = "#D55E00"
PURPLE = "#CC79A7"

MODEL_DISPLAY = {
    "lab_ordering_only":    ("Lab ordering\n(no values)",      SKY,    None),
    "imaging_ordering":     ("Imaging ordering\n(no values)",  ORANGE, None),
    "med_class_ordering":   ("Med class ordering\n(no values)",PURPLE, None),
    "full_behavioral":      ("Full behavioral\n(all ordering)", BLUE,  "///"),
    "values_only":          ("Values only\n(lab results)",      GREEN,  None),
    "behavioral_plus_values":("Behavioral\n+ values",           RED,   "///"),
}

with open(OUT / "behavioral_results.json") as f:
    results = json.load(f)

print("=" * 75)
print("BEHAVIORAL FINGERPRINT — RESULTS")
print("=" * 75)
for task in TASKS:
    r = results[task]
    print(f"\n  {task}  (N={r.get('N_test', ''):,}, prev={r.get('prevalence_test', 0):.1%})")
    for m, (label, color, _) in MODEL_DISPLAY.items():
        if m not in r:
            continue
        auc, lo, hi = r[m]["auroc"], r[m]["ci_lo"], r[m]["ci_hi"]
        print(f"    {m:<28}: {auc:.4f} [{lo:.4f}–{hi:.4f}]")

print()
print("KEY FINDING — Full behavioral vs values-only:")
for task in TASKS:
    r = results[task]
    fb = r.get("full_behavioral", {}).get("auroc")
    vo = r.get("values_only", {}).get("auroc")
    if fb and vo:
        print(f"  {task}: Full-beh {fb:.4f}  vs  Values {vo:.4f}  (Δ={fb - vo:+.4f})")
print("=" * 75)

# ── Figure ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 7.5, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle(
    "Figure 14 — Behavioral Fingerprint: Extending the Ordering Signal\n"
    "Lab ordering vs imaging ordering vs medication class ordering vs combined",
    fontsize=11, fontweight="bold", y=1.03
)

x = np.arange(len(MODEL_DISPLAY))
w = 0.6

for ax_idx, task in enumerate(TASKS):
    ax = axes[ax_idx]
    r  = results[task]

    labels  = []
    aurocs  = []
    lo_errs = []
    hi_errs = []
    colors  = []
    hatches = []

    for m, (lbl, col, hatch) in MODEL_DISPLAY.items():
        if m not in r:
            continue
        labels.append(lbl)
        aurocs.append(r[m]["auroc"])
        lo_errs.append(r[m]["auroc"] - r[m]["ci_lo"])
        hi_errs.append(r[m]["ci_hi"] - r[m]["auroc"])
        colors.append(col)
        hatches.append(hatch)

    xi = np.arange(len(aurocs))
    bars = ax.bar(xi, aurocs, width=w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for b, hatch in zip(bars, hatches):
        if hatch:
            b.set_hatch(hatch)

    ax.errorbar(xi, aurocs, yerr=[lo_errs, hi_errs],
                fmt="none", color="#333333", capsize=3.5, linewidth=1.2)

    for i, (a, he) in enumerate(zip(aurocs, hi_errs)):
        ax.text(i, a + he + 0.003, f"{a:.3f}",
                ha="center", fontsize=7.5, fontweight="bold", color="#333333")

    ymin = max(0.50, min(aurocs) - 0.06)
    ymax = max(aurocs) + max(hi_errs) + 0.06
    ax.set_ylim(ymin, min(ymax, 1.0))
    ax.set_xticks(xi)
    ax.set_xticklabels(labels, fontsize=7.5, ha="center")
    ax.set_ylabel("AUROC" if ax_idx == 0 else "")
    ax.set_title(TASK_LABELS[task])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.text(0.98, 0.03,
            f"N={r.get('N_test', 0):,}\nPrev={r.get('prevalence_test', 0):.1%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")

patches = [mpatches.Patch(color=col, alpha=0.85, label=lbl)
           for _, (lbl, col, _) in MODEL_DISPLAY.items()]
fig.legend(handles=patches, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig(FIG / "fig14_behavioral.png", bbox_inches="tight")
plt.close()
print("Saved → fig14_behavioral.png")
