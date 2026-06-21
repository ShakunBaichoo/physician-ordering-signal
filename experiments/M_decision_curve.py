#!/usr/bin/env python
"""
M_decision_curve.py
====================
Decision Curve Analysis (DCA) for ordering-only, values-only, and combined
LightGBM models across four clinical outcomes.

DCA quantifies net clinical benefit at every probability threshold:
  net_benefit(t) = TPR - FPR × t/(1-t)

where t is the decision threshold at which a clinician would act.
Compared to "treat-all" (order intervention for everyone) and
"treat-none" (order intervention for no one).

Outputs → 1_ordering_paper/results/M_dca/
  dca_results.json        — peak net benefit, threshold ranges, summaries
  dca_curves.csv          — full curve data for all models × tasks
  fig12_dca.png           — 4-panel figure (one panel per task)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
PRED = ROOT / "1_ordering_paper" / "results" / "H_cci_stratified" / "test_predictions.parquet"
OUT  = ROOT / "1_ordering_paper" / "results" / "M_dca"
FIG  = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# ── Wong (2011) colorblind-safe palette ───────────────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "aki":             "Acute Kidney Injury",
    "sepsis":          "Sepsis",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  8.5,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


# ── Core DCA function ─────────────────────────────────────────────────────────

def compute_net_benefit(y_true: np.ndarray, y_prob: np.ndarray,
                        thresholds: np.ndarray) -> np.ndarray:
    """
    net_benefit(t) = (TP/N) - (FP/N) × t/(1-t)

    Returns NaN for thresholds where 1-t ≈ 0 to avoid division errors.
    """
    N = len(y_true)
    nb = np.full(len(thresholds), np.nan)
    for i, t in enumerate(thresholds):
        if t >= 1.0:
            continue
        preds = (y_prob >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb[i] = (tp / N) - (fp / N) * (t / (1.0 - t))
    return nb


def treat_all_nb(prevalence: float, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit if every patient receives the intervention."""
    nb = np.full(len(thresholds), np.nan)
    for i, t in enumerate(thresholds):
        if t >= 1.0:
            continue
        nb[i] = prevalence - (1.0 - prevalence) * (t / (1.0 - t))
    return nb


# ── Load predictions ──────────────────────────────────────────────────────────
print("Loading test predictions...")
df = pd.read_parquet(PRED)
print(f"  {len(df):,} rows × {df.shape[1]} cols | tasks: {df['task'].unique().tolist()}")


# ── Threshold grid ────────────────────────────────────────────────────────────
# Clinical decision thresholds: 1% – 60%
# Finer resolution at low thresholds (where most action happens)
THRESHOLDS = np.concatenate([
    np.arange(0.01, 0.10, 0.002),
    np.arange(0.10, 0.30, 0.005),
    np.arange(0.30, 0.61, 0.01),
])


# ── Compute DCA for all tasks + models ───────────────────────────────────────
results = {}
curves_rows = []

for task in TASKS:
    sub = df[df["task"] == task].dropna(subset=["true_label", "ord_prob", "val_prob", "both_prob"])
    y    = sub["true_label"].values.astype(float)
    prev = y.mean()
    N    = len(y)

    nb_ord  = compute_net_benefit(y, sub["ord_prob"].values,  THRESHOLDS)
    nb_val  = compute_net_benefit(y, sub["val_prob"].values,  THRESHOLDS)
    nb_both = compute_net_benefit(y, sub["both_prob"].values, THRESHOLDS)
    nb_all  = treat_all_nb(prev, THRESHOLDS)
    nb_none = np.zeros(len(THRESHOLDS))

    # Clip very negative values for readability (below −0.05 is practically useless)
    clip = -0.05
    nb_ord  = np.clip(nb_ord,  clip, None)
    nb_val  = np.clip(nb_val,  clip, None)
    nb_both = np.clip(nb_both, clip, None)
    nb_all  = np.clip(nb_all,  clip, None)

    # --- Summaries ---
    # Net benefit range where ordering has positive benefit (above treat-none=0)
    pos_ord  = THRESHOLDS[(nb_ord  > 0) & (~np.isnan(nb_ord))]
    pos_val  = THRESHOLDS[(nb_val  > 0) & (~np.isnan(nb_val))]
    pos_both = THRESHOLDS[(nb_both > 0) & (~np.isnan(nb_both))]

    # Threshold range where ordering ≈ values (within 0.005 net benefit)
    close_mask = (~np.isnan(nb_ord)) & (~np.isnan(nb_val)) & (np.abs(nb_ord - nb_val) <= 0.005)
    close_thresholds = THRESHOLDS[close_mask]

    # Peak net benefit
    peak_ord  = float(np.nanmax(nb_ord))  if len(nb_ord) else np.nan
    peak_val  = float(np.nanmax(nb_val))  if len(nb_val) else np.nan
    peak_both = float(np.nanmax(nb_both)) if len(nb_both) else np.nan

    results[task] = {
        "N": int(N),
        "prevalence": round(float(prev), 4),
        "peak_net_benefit": {
            "ordering_only": round(peak_ord, 4),
            "values_only":   round(peak_val, 4),
            "combined":      round(peak_both, 4),
        },
        "positive_nb_threshold_range": {
            "ordering_only": [round(float(pos_ord.min()), 3), round(float(pos_ord.max()), 3)] if len(pos_ord) else [],
            "values_only":   [round(float(pos_val.min()), 3), round(float(pos_val.max()), 3)] if len(pos_val) else [],
            "combined":      [round(float(pos_both.min()), 3), round(float(pos_both.max()), 3)] if len(pos_both) else [],
        },
        "ordering_equiv_values_thresholds": (
            [round(float(close_thresholds.min()), 3), round(float(close_thresholds.max()), 3)]
            if len(close_thresholds) > 0 else []
        ),
    }
    print(f"  {task}: prevalence={prev:.3f}, peak NB ordering={peak_ord:.4f}, "
          f"values={peak_val:.4f}, combined={peak_both:.4f}")
    print(f"    Positive NB range (ordering): {results[task]['positive_nb_threshold_range']['ordering_only']}")

    # Save curves for CSV
    for t, no, nv, nb, na in zip(THRESHOLDS, nb_ord, nb_val, nb_both, nb_all):
        curves_rows.append({
            "task": task, "threshold": round(t, 4),
            "ordering_only": round(no, 5) if not np.isnan(no) else np.nan,
            "values_only":   round(nv, 5) if not np.isnan(nv) else np.nan,
            "combined":      round(nb, 5) if not np.isnan(nb) else np.nan,
            "treat_all":     round(na, 5) if not np.isnan(na) else np.nan,
            "treat_none":    0.0,
        })


# ── Save outputs ──────────────────────────────────────────────────────────────
with open(OUT / "dca_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved dca_results.json")

curves_df = pd.DataFrame(curves_rows)
curves_df.to_csv(OUT / "dca_curves.csv", index=False)
print(f"Saved dca_curves.csv ({len(curves_df):,} rows)")


# ── Figure ────────────────────────────────────────────────────────────────────
print("\nBuilding Figure 12 — Decision Curve Analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

MODEL_STYLES = [
    ("ordering_only", "Ordering-only",  BLUE,   "-",  2.0),
    ("values_only",   "Values-only",    ORANGE, "-",  2.0),
    ("combined",      "Combined",       GREEN,  "-",  2.0),
    ("treat_all",     "Treat all",      "#888888", "--", 1.2),
]

for ax_idx, task in enumerate(TASKS):
    ax   = axes[ax_idx]
    sub  = curves_df[curves_df["task"] == task]
    prev = results[task]["prevalence"]
    N    = results[task]["N"]

    # Shade region where ordering has positive net benefit
    ord_pos = sub[sub["ordering_only"] > 0]
    if len(ord_pos) > 0:
        ax.axvspan(ord_pos["threshold"].min(), ord_pos["threshold"].max(),
                   alpha=0.07, color=BLUE, label="_nolegend_")

    # Plot treat-none at zero
    ax.axhline(0, color="#999999", linewidth=0.9, linestyle=":", zorder=1,
               label="Treat none")

    for col, label, color, ls, lw in MODEL_STYLES:
        y = sub[col].values
        x = sub["threshold"].values
        mask = ~np.isnan(y)
        ax.plot(x[mask], y[mask], color=color, linestyle=ls,
                linewidth=lw, label=label, zorder=3)

    # Mark peak net benefit for ordering
    peak_idx = np.nanargmax(sub["ordering_only"].values)
    peak_t   = sub["threshold"].iloc[peak_idx]
    peak_nb  = sub["ordering_only"].iloc[peak_idx]
    ax.scatter([peak_t], [peak_nb], color=BLUE, s=40, zorder=5,
               marker="*", linewidths=0)
    ax.annotate(f"NB={peak_nb:.3f}\n@t={peak_t:.2f}",
                xy=(peak_t, peak_nb), xytext=(peak_t + 0.03, peak_nb + 0.003),
                fontsize=7.5, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8))

    # Annotate the "threshold range with positive NB"
    pos_range = results[task]["positive_nb_threshold_range"]["ordering_only"]
    if len(pos_range) == 2:
        ax.text(0.97, 0.97,
                f"Positive NB: {pos_range[0]:.0%}–{pos_range[1]:.0%}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=BLUE,
                bbox=dict(boxstyle="round,pad=0.3", fc="#EAF4FB", ec=BLUE, alpha=0.8))

    ax.set_title(TASK_LABELS[task], pad=8)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Net benefit" if ax_idx % 2 == 0 else "")
    ax.set_xlim(0.01, 0.60)

    # Y-axis: just above the treat-all line at threshold 0.01
    ymin = -0.03
    ymax = max(
        float(np.nanmax(sub["ordering_only"].values)),
        float(np.nanmax(sub["values_only"].values)),
        float(np.nanmax(sub["treat_all"].values)),
    ) * 1.20
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color="#cccccc", linewidth=0.5)

    # Prevalence annotation
    ax.text(0.02, 0.03,
            f"Prevalence: {prev:.1%}  |  N={N:,}",
            transform=ax.transAxes, fontsize=7.5, color="#555555")

    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    if ax_idx == 0:
        ax.legend(frameon=False, loc="upper right", fontsize=8.5)

fig.suptitle(
    "Figure 12 — Decision Curve Analysis: Net Clinical Benefit of Ordering-Only Model\n"
    "Shaded region = threshold range where ordering-only achieves positive net benefit\n"
    "★ = peak net benefit for ordering-only model",
    fontsize=11, fontweight="bold", y=1.01
)

plt.tight_layout()
plt.savefig(FIG / "fig12_dca.png", bbox_inches="tight")
plt.close()
print("  → fig12_dca.png")


# ── Print summary table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("DECISION CURVE ANALYSIS — SUMMARY")
print("="*70)
print(f"{'Task':<18} {'Prev':>6} {'Peak NB (ord)':>14} {'Peak NB (val)':>14} "
      f"{'Peak NB (comb)':>15} {'Positive NB range':>20}")
print("-"*87)
for task in TASKS:
    r = results[task]
    p = r["peak_net_benefit"]
    rng = r["positive_nb_threshold_range"]["ordering_only"]
    rng_str = f"{rng[0]:.0%}–{rng[1]:.0%}" if len(rng) == 2 else "—"
    print(f"{task:<18} {r['prevalence']:>6.1%} {p['ordering_only']:>14.4f} "
          f"{p['values_only']:>14.4f} {p['combined']:>15.4f} {rng_str:>20}")
print("="*70)
print(f"\nAll DCA outputs → {OUT}")
print(f"Figure 12       → {FIG / 'fig12_dca.png'}")
