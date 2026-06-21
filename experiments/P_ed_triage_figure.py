#!/usr/bin/env python
"""Regenerate figure and summary for Experiment P from saved results."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

ROOT   = Path(__file__).parents[2]
OUT    = ROOT / "1_ordering_paper" / "results" / "P_ed_triage"
FIG    = ROOT / "1_ordering_paper" / "results" / "figures"

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "aki":             "AKI",
    "sepsis":          "Sepsis",
}
MODEL_DISPLAY = {
    "triage_vitals_only": ("Triage vitals\n(HR/RR/SpO₂/SBP…)", "#56B4E9", None),
    "ed_lab_ordering":    ("ED lab ordering\n(no values)",       "#0072B2", "///"),
    "ed_full_behavioral": ("ED full behavioral\n(labs+img+meds)", "#009E73", "///"),
}

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
SKY    = "#56B4E9"

with open(OUT / "ed_triage_results.json") as f:
    all_results = json.load(f)

# Load reference AUROCs
RAND_CI  = ROOT / "1_ordering_paper" / "results" / "F_paper_improvements" / "bootstrap_ci_results.json"
L_TRIAGE = ROOT / "1_ordering_paper" / "results" / "L_triage_window" / "L_triage_summary.json"

try:
    with open(RAND_CI) as f:
        rand_ci = json.load(f)
except:
    rand_ci = {}

# Build lookup: task → model → auroc from L triage list
l_lookup = {}
try:
    with open(L_TRIAGE) as f:
        l_data = json.load(f)
    for rec in l_data.get("triage_auroc", []):
        t, m = rec["task"], rec["model"]
        if t not in l_lookup:
            l_lookup[t] = {}
        l_lookup[t][m] = rec["auroc"]
except:
    pass

# ── Summary table ─────────────────────────────────────────────────────────────
print("="*75)
print("ED TRIAGE VALIDATION — RESULTS")
print("="*75)
for task in TASKS:
    r = all_results[task]
    print(f"\n  {task}  (N={r.get('N_test',''):,}, prev={r.get('prevalence_test',0):.1%})")
    for name in MODEL_DISPLAY:
        if name not in r: continue
        auc, lo, hi = r[name]["auroc"], r[name]["ci_lo"], r[name]["ci_hi"]
        print(f"    {name:<24}: {auc:.4f} [{lo:.4f}–{hi:.4f}]")
    inp = rand_ci.get(task, {}).get("ordering_only", {}).get("auroc")
    l_t = l_lookup.get(task, {}).get("ordering_only")
    if inp: print(f"    {'inpatient ord (ref)':<24}: {inp:.4f}")
    if l_t: print(f"    {'inpatient triage t=0-4h':<24}: {l_t:.4f}")

print()
print("KEY FINDING — ED ordering vs inpatient ordering-only:")
for task in TASKS:
    r     = all_results[task]
    ed_fb = r.get("ed_full_behavioral", {}).get("auroc")
    inp   = rand_ci.get(task, {}).get("ordering_only", {}).get("auroc")
    if ed_fb and inp:
        print(f"  {task}: ED {ed_fb:.4f}  vs  inpatient {inp:.4f}  (Δ={ed_fb-inp:+.4f})")
print("="*75)

# ── Figure ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
fig.suptitle(
    "Figure 15 — Emergency Department Triage Validation\n"
    "Predicting inpatient outcomes from ED ordering behavior "
    "before any lab results are available",
    fontsize=11, fontweight="bold", y=1.03
)

x = np.arange(len(MODEL_DISPLAY))
w = 0.55

for ax_idx, task in enumerate(TASKS):
    ax = axes[ax_idx]
    r  = all_results[task]

    aurocs  = [r[m]["auroc"] for m in MODEL_DISPLAY if m in r]
    lo_errs = [r[m]["auroc"] - r[m]["ci_lo"] for m in MODEL_DISPLAY if m in r]
    hi_errs = [r[m]["ci_hi"] - r[m]["auroc"] for m in MODEL_DISPLAY if m in r]
    colors  = [c for _, c, _ in MODEL_DISPLAY.values()]
    hatches = [h for _, _, h in MODEL_DISPLAY.values()]

    bars = ax.bar(x, aurocs, width=w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for b, hatch in zip(bars, hatches):
        if hatch:
            b.set_hatch(hatch)
    ax.errorbar(x, aurocs, yerr=[lo_errs, hi_errs],
                fmt="none", color="#333333", capsize=4, linewidth=1.3)

    for xi, (a, he) in enumerate(zip(aurocs, hi_errs)):
        ax.text(xi, a + he + 0.004, f"{a:.3f}",
                ha="center", fontsize=8, fontweight="bold", color="#333333")

    inp_ref = rand_ci.get(task, {}).get("ordering_only", {}).get("auroc")
    l_ref   = l_lookup.get(task, {}).get("ordering_only")

    if isinstance(inp_ref, float):
        ax.axhline(inp_ref, color=BLUE, linewidth=1.5, linestyle="--", alpha=0.6)
        ax.text(len(x) - 0.3, inp_ref + 0.003,
                f"IP ord.\n{inp_ref:.3f}", fontsize=6.5, color=BLUE,
                ha="right", va="bottom")
    if isinstance(l_ref, float):
        ax.axhline(l_ref, color=ORANGE, linewidth=1.5, linestyle=":", alpha=0.7)
        ax.text(len(x) - 0.3, l_ref - 0.013,
                f"IP triage\n{l_ref:.3f}", fontsize=6.5, color=ORANGE,
                ha="right", va="top")

    ymin = max(0.50, min(aurocs) - 0.06)
    ymax = max(aurocs) + max(hi_errs) + 0.06
    if isinstance(inp_ref, float): ymax = max(ymax, inp_ref + 0.04)
    ax.set_ylim(ymin, min(ymax, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in MODEL_DISPLAY.values()],
                       fontsize=8, ha="center")
    ax.set_ylabel("AUROC" if ax_idx == 0 else "")
    ax.set_title(TASK_LABELS[task])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.text(0.98, 0.03,
            f"N={r.get('N_test',0):,}\nPrev={r.get('prevalence_test',0):.1%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")

patches = [mpatches.Patch(color=c, alpha=0.85, label=l)
           for _, (l, c, _) in MODEL_DISPLAY.items()]
patches += [
    Line2D([0],[0], color=BLUE,   lw=1.5, ls="--", alpha=0.6,
           label="Inpatient ordering-only (reference)"),
    Line2D([0],[0], color=ORANGE, lw=1.5, ls=":",  alpha=0.7,
           label="Inpatient triage t=0–4h (reference)"),
]
fig.legend(handles=patches, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig(FIG / "fig15_ed_triage.png", bbox_inches="tight")
plt.close()
print("Saved → fig15_ed_triage.png")
