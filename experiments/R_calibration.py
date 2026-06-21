#!/usr/bin/env python
"""
Experiment R — Calibration Analysis
=====================================
For each model (ordering-only, values-only, combined) × 4 tasks:
  - Reliability diagram (15 bins)
  - Expected Calibration Error (ECE) before and after isotonic regression
  - Brier score
Quantifies whether the predicted probabilities are trustworthy at the operating
thresholds reported in the main analyses.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).parents[2]
OUT  = ROOT / "1_ordering_paper" / "results" / "R_calibration"
FIG  = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PRED = ROOT / "1_ordering_paper" / "results" / "H_cci_stratified" / "test_predictions.parquet"

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
TASK_LABELS = {
    "mortality":       "In-Hospital\nMortality",
    "readmission_30d": "30-Day\nReadmission",
    "aki":             "AKI",
    "sepsis":          "Sepsis",
}
MODELS = {
    "ord_prob":  ("Ordering only",  "#0072B2"),
    "val_prob":  ("Values only",    "#009E73"),
    "both_prob": ("Combined",       "#D55E00"),
}
N_BINS = 15


def ece(y_true, y_prob, n_bins=N_BINS):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_prob = y_prob[mask].mean()
        ece_val += (mask.sum() / n) * abs(frac_pos - mean_prob)
    return ece_val


def calibrate_isotonic(y_true_tr, y_prob_tr, y_prob_te):
    """Fit isotonic regression on first half, apply to second half."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob_tr, y_true_tr)
    return ir.predict(y_prob_te)


# ── Load predictions ────────────────────────────────────────────────────────
df = pd.read_parquet(PRED)
print(f"Loaded {len(df):,} test predictions")

results = {}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(18, 14))
fig.suptitle(
    "Figure 17 — Calibration Analysis: Are Predicted Probabilities Trustworthy?\n"
    "Reliability diagrams before and after isotonic recalibration (N=45,805 test admissions)",
    fontsize=11, fontweight="bold", y=1.01
)

# 4 tasks × 3 models = 12 subplots, arranged 4 cols × 3 rows
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.35)

for col_idx, task in enumerate(TASKS):
    task_df = df[df["task"] == task].dropna(subset=["true_label"]).copy()
    y = task_df["true_label"].values.astype(int)
    prev = task_df["true_label"].mean()
    n    = len(task_df)

    task_res = {"N": int(n), "prevalence": float(prev)}

    # Split in half: first half for calibration fitting, second for evaluation
    half = n // 2
    y_tr, y_te = y[:half], y[half:]

    for row_idx, (col, (label, color)) in enumerate(MODELS.items()):
        ax = fig.add_subplot(gs[row_idx, col_idx])
        p    = task_df[col].values
        p_tr, p_te = p[:half], p[half:]

        # Pre-calibration on test half
        frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=N_BINS,
                                                 strategy="uniform")
        ece_pre   = ece(y_te, p_te)
        brier_pre = brier_score_loss(y_te, p_te)
        auc_pre   = roc_auc_score(y_te, p_te)

        # Isotonic recalibration: fit on train half, evaluate on test half
        p_cal = calibrate_isotonic(y_tr, p_tr, p_te)
        frac_pos_cal, mean_pred_cal = calibration_curve(y_te, p_cal, n_bins=N_BINS,
                                                         strategy="uniform")
        ece_post   = ece(y_te, p_cal)
        brier_post = brier_score_loss(y_te, p_cal)

        # Plot
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Perfect")
        ax.plot(mean_pred, frac_pos, "o-", color=color, markersize=4,
                linewidth=1.5, label=f"Raw (ECE={ece_pre:.3f})", alpha=0.9)
        ax.plot(mean_pred_cal, frac_pos_cal, "s--", color=color, markersize=4,
                linewidth=1.5, alpha=0.6,
                label=f"Calibrated (ECE={ece_post:.3f})")

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability")
        if col_idx == 0:
            ax.set_ylabel(f"{label}\nFraction positives")
        if row_idx == 0:
            ax.set_title(TASK_LABELS[task])
        ax.legend(fontsize=6.5, loc="upper left", frameon=False)
        ax.text(0.98, 0.05,
                f"Brier: {brier_pre:.4f}→{brier_post:.4f}\nAUROC: {auc_pre:.4f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color="#555555")

        task_res[col] = {
            "ece_pre":    round(ece_pre, 4),
            "ece_post":   round(ece_post, 4),
            "ece_reduction_pct": round((ece_pre - ece_post) / ece_pre * 100, 1),
            "brier_pre":  round(brier_pre, 4),
            "brier_post": round(brier_post, 4),
            "auroc":      round(auc_pre, 4),
        }

    results[task] = task_res

plt.savefig(FIG / "fig17_calibration.png", bbox_inches="tight")
plt.close()
print("Saved → fig17_calibration.png")

# ── Summary table ────────────────────────────────────────────────────────────
with open(OUT / "calibration_results.json", "w") as f:
    json.dump(results, f, indent=2)

rows = []
for task, r in results.items():
    for m, (lbl, _) in MODELS.items():
        v = r.get(m, {})
        rows.append({
            "task": task, "model": lbl,
            "ece_pre": v.get("ece_pre"), "ece_post": v.get("ece_post"),
            "ece_reduction_pct": v.get("ece_reduction_pct"),
            "brier_pre": v.get("brier_pre"), "brier_post": v.get("brier_post"),
            "auroc": v.get("auroc"),
        })
pd.DataFrame(rows).to_csv(OUT / "calibration_results.csv", index=False)

print("\n" + "=" * 75)
print("CALIBRATION SUMMARY")
print("=" * 75)
print(f"{'Task':<18} {'Model':<20} {'ECE (pre)':>9} {'ECE (post)':>10} {'Reduction':>10} {'Brier':>8}")
print("-" * 75)
for task, r in results.items():
    for m, (lbl, _) in MODELS.items():
        v = r.get(m, {})
        print(f"  {task:<16} {lbl:<20} {v.get('ece_pre',0):>9.4f} "
              f"{v.get('ece_post',0):>10.4f} {v.get('ece_reduction_pct',0):>9.1f}% "
              f"{v.get('brier_pre',0):>8.4f}")
print("=" * 75)
print("\nKey: ECE = Expected Calibration Error (lower = better calibrated)")
print("     Isotonic recalibration applied: train-half fit → test-half evaluate")
