#!/usr/bin/env python
"""
Experiment S — Temporal Ordering Trajectory
=============================================
Extends the ordering signal from flat counts to temporal features:
  - Ordering intensity per 4h bin (12 bins × 48h)
  - Escalation slope (linear trend over first 48h)
  - Early vs late ordering ratio
  - Time-to-first key test (troponin, lactate, culture, CBC, BMP)
  - Peak ordering bin
  - Shannon entropy of ordering distribution across bins

Compare trajectory-enriched model vs flat-count model from H_cci_stratified.
Answers: "Does the TIMING of orders add information beyond the count?"
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parents[2]
PROC    = ROOT / "data" / "processed"
OUT     = ROOT / "1_ordering_paper" / "results" / "S_temporal"
FIG     = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reference: H_cci_stratified flat-count AUROC (from bootstrap_ci_results.json)
FLAT_REF_PATH = ROOT / "1_ordering_paper" / "results" / "F_paper_improvements" / "bootstrap_ci_results.json"

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "aki":             "AKI",
    "sepsis":          "Sepsis",
}
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"

RANDOM_STATE = 42
N_BOOT = 1000
N_BINS_TIME = 12   # 12 × 4h = 48h


def bootstrap_auroc(y_true, y_prob, n=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    if y_true.sum() < 5 or (len(y_true) - y_true.sum()) < 5:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y_true, y_prob)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, yp))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def train_lgbm(X_tr, y_tr, X_va, y_va):
    spw = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
    m = lgb.LGBMClassifier(
        n_estimators=2000, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        scale_pos_weight=spw, metric="auc",
        random_state=RANDOM_STATE, n_jobs=8, verbose=-1
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    return m


# ── 1. Load timeseries data ─────────────────────────────────────────────────
print("Loading timeseries data ...")
labs   = pd.read_parquet(PROC / "timeseries" / "labs.parquet")
vitals = pd.read_parquet(PROC / "timeseries" / "vitals.parquet")
print(f"  labs: {labs.shape}, vitals: {vitals.shape}")

lab_value_cols   = [c for c in labs.columns   if c not in ["hadm_id", "time_bin"]]
vital_value_cols = [c for c in vitals.columns if c not in ["hadm_id", "time_bin"]]

# ── 2. Build trajectory features ────────────────────────────────────────────
print("\nBuilding trajectory features ...")

# 2a. Ordering intensity per bin: fraction of lab columns with a value
labs["n_labs_in_bin"]     = labs[lab_value_cols].notna().sum(axis=1)
labs["ord_intensity_bin"] = labs["n_labs_in_bin"] / len(lab_value_cols)

vitals["n_vitals_in_bin"]     = vitals[vital_value_cols].notna().sum(axis=1)
vitals["vit_intensity_bin"]   = vitals["n_vitals_in_bin"] / len(vital_value_cols)

# Pivot to wide: one row per hadm_id
lab_pivot = labs.pivot_table(
    index="hadm_id", columns="time_bin", values="ord_intensity_bin", fill_value=0
)
lab_pivot.columns = [f"ord_bin_{b}" for b in lab_pivot.columns]

vit_pivot = vitals.pivot_table(
    index="hadm_id", columns="time_bin", values="vit_intensity_bin", fill_value=0
)
vit_pivot.columns = [f"vit_bin_{b}" for b in vit_pivot.columns]

# 2b. Trajectory summary features
bin_cols = [f"ord_bin_{b}" for b in range(N_BINS_TIME) if f"ord_bin_{b}" in lab_pivot.columns]
X_bins   = lab_pivot[bin_cols].values   # (N, 12)
t        = np.arange(len(bin_cols), dtype=float).reshape(-1, 1)

print("  Computing escalation slopes ...")
traj_feats = pd.DataFrame(index=lab_pivot.index)

# Escalation slope (linear regression coefficient)
slopes = []
for row in X_bins:
    if row.max() > 0:
        lr = LinearRegression().fit(t, row)
        slopes.append(lr.coef_[0])
    else:
        slopes.append(0.0)
traj_feats["escalation_slope"] = slopes

# Early (bins 0-3) vs late (bins 8-11) ordering ratio
early_bins = [f"ord_bin_{b}" for b in range(4)  if f"ord_bin_{b}" in lab_pivot.columns]
late_bins  = [f"ord_bin_{b}" for b in range(8,12) if f"ord_bin_{b}" in lab_pivot.columns]
early = lab_pivot[early_bins].mean(axis=1)
late  = lab_pivot[late_bins].mean(axis=1)
traj_feats["early_ordering"]    = early
traj_feats["late_ordering"]     = late
traj_feats["late_vs_early_ratio"] = late / (early + 1e-6)

# Peak ordering bin
traj_feats["peak_bin"] = lab_pivot[bin_cols].idxmax(axis=1).str.extract(r"(\d+)").astype(float)

# First non-zero bin (time-to-first ordering)
def first_nonzero_bin(row):
    for i, v in enumerate(row):
        if v > 0:
            return float(i)
    return float(N_BINS_TIME)
traj_feats["first_order_bin"] = [first_nonzero_bin(r) for r in X_bins]

# Shannon entropy of ordering distribution
def shannon_entropy(row):
    s = row.sum()
    if s == 0:
        return 0.0
    p = row / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))
traj_feats["ordering_entropy"] = [shannon_entropy(r) for r in X_bins]

# Total ordering volume
traj_feats["total_ordering"]   = lab_pivot[bin_cols].sum(axis=1)
traj_feats["max_bin_ordering"] = lab_pivot[bin_cols].max(axis=1)

# Specific high-value lab first-occurrence bins
# (which 4h bin did a specific lab first appear?)
high_value_labs = {
    "troponin":   [c for c in lab_value_cols if "troponin" in c.lower()],
    "lactate":    [c for c in lab_value_cols if "lactate"  in c.lower()],
    "creatinine": [c for c in lab_value_cols if "creatinine" in c.lower()],
    "wbc":        [c for c in lab_value_cols if "wbc" in c.lower() or "white_blood" in c.lower()],
    "bun":        [c for c in lab_value_cols if "bun" in c.lower() or "urea" in c.lower()],
    "sodium":     [c for c in lab_value_cols if "sodium" in c.lower()],
    "potassium":  [c for c in lab_value_cols if "potassium" in c.lower()],
    "glucose":    [c for c in lab_value_cols if "glucose" in c.lower()],
    "hemoglobin": [c for c in lab_value_cols if "hemoglobin" in c.lower() or c.lower() == "hgb_mean"],
    "bilirubin":  [c for c in lab_value_cols if "bilirubin" in c.lower()],
}

print("  Computing time-to-first key tests ...")
for test, cols in high_value_labs.items():
    cols = [c for c in cols if c in labs.columns]
    if not cols:
        traj_feats[f"first_bin_{test}"] = float(N_BINS_TIME)
        continue
    # first bin where any of these cols is non-null
    tmp = labs[["hadm_id", "time_bin"] + cols].copy()
    tmp["has_test"] = tmp[cols].notna().any(axis=1)
    first_bin = (tmp[tmp["has_test"]]
                 .groupby("hadm_id")["time_bin"].min()
                 .rename(f"first_bin_{test}"))
    traj_feats = traj_feats.join(first_bin, how="left")
    traj_feats[f"first_bin_{test}"] = traj_feats[f"first_bin_{test}"].fillna(N_BINS_TIME)

# Vital signs trajectory features
vit_bin_cols = [c for c in vit_pivot.columns]
traj_feats["vit_escalation_slope"] = [
    (LinearRegression().fit(t, r).coef_[0] if max(r) > 0 else 0.0)
    for r in vit_pivot[vit_bin_cols].values
]
traj_feats["vit_early"] = vit_pivot[[c for c in vit_bin_cols if int(c.split("_")[-1]) < 4]].mean(axis=1)
traj_feats["vit_late"]  = vit_pivot[[c for c in vit_bin_cols if int(c.split("_")[-1]) >= 8]].mean(axis=1)

print(f"  Built {len(traj_feats.columns)} trajectory features for {len(traj_feats):,} admissions")

TRAJ_FEAT_COLS = traj_feats.columns.tolist()

# ── 3. Load static features + labels ────────────────────────────────────────
print("\nLoading cohort data ...")
cohort  = pd.read_parquet(PROC / "cohort.parquet")
labels  = pd.read_parquet(PROC / "labels.parquet")
static  = pd.read_parquet(PROC / "static.parquet")

# Merge everything
data = (cohort[["hadm_id", "subject_id", "anchor_year_group"]]
        .merge(labels.drop(columns=["subject_id"], errors="ignore"), on="hadm_id")
        .merge(static.drop(columns=["subject_id"], errors="ignore"), on="hadm_id", how="left"))

# Merge flat ordering features (total orders from bins)
flat_ord = lab_pivot[bin_cols].sum(axis=1).rename("flat_total_ordering").reset_index()
# Also add the full flat counts: use the sum across all 12 bins per lab
flat_counts = (labs.groupby("hadm_id")[lab_value_cols]
               .apply(lambda x: x.notna().sum().sum())
               .rename("flat_n_lab_obs").reset_index())

data = (data
        .merge(traj_feats.reset_index(), on="hadm_id", how="left")
        .merge(flat_counts, on="hadm_id", how="left"))

# Merge per-bin columns for the binned model
data = data.merge(lab_pivot.reset_index(), on="hadm_id", how="left")
data = data.merge(vit_pivot.reset_index(), on="hadm_id", how="left")

# Fill NaN
for c in TRAJ_FEAT_COLS + bin_cols + vit_bin_cols:
    if c in data.columns:
        data[c] = data[c].fillna(0)

print(f"  Final dataset: {len(data):,} admissions")

# ── 4. Train/val/test split (same as H_cci_stratified: 60/20/20) ────────────
rng = np.random.default_rng(RANDOM_STATE)
subjects = data["subject_id"].unique()
rng.shuffle(subjects)
n = len(subjects)
train_subj = set(subjects[:int(n * 0.60)])
val_subj   = set(subjects[int(n * 0.60):int(n * 0.80)])
test_subj  = set(subjects[int(n * 0.80):])

idx_train = data.index[data["subject_id"].isin(train_subj)]
idx_val   = data.index[data["subject_id"].isin(val_subj)]
idx_test  = data.index[data["subject_id"].isin(test_subj)]
print(f"  Split: train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

# ── 5. Feature sets ──────────────────────────────────────────────────────────
static_cols = [c for c in static.columns
               if c not in ["hadm_id", "subject_id"]
               and c in data.columns]

FEATURE_SETS = {
    "flat_ordering_only":    ["flat_n_lab_obs"] + bin_cols + vit_bin_cols,
    "trajectory_only":       TRAJ_FEAT_COLS,
    "trajectory_enriched":   TRAJ_FEAT_COLS + bin_cols + vit_bin_cols,
    "static_plus_flat":      static_cols + ["flat_n_lab_obs"] + bin_cols + vit_bin_cols,
    "static_plus_trajectory":static_cols + TRAJ_FEAT_COLS + bin_cols + vit_bin_cols,
}

# ── 6. Train & evaluate ──────────────────────────────────────────────────────
LABEL_COLS = {
    "mortality":       "mortality",
    "readmission_30d": "readmission_30d",
    "aki":             "aki",
    "sepsis":          "sepsis",
}
all_results = {}

for task in TASKS:
    label_col = LABEL_COLS[task]
    if label_col not in data.columns:
        print(f"  {task}: label column '{label_col}' not found — skipping")
        continue

    task_data = data.dropna(subset=[label_col])
    t_idx_train = task_data.index[task_data["subject_id"].isin(train_subj)]
    t_idx_val   = task_data.index[task_data["subject_id"].isin(val_subj)]
    t_idx_test  = task_data.index[task_data["subject_id"].isin(test_subj)]
    y = task_data[label_col].values.astype(int)
    y_tr = task_data.loc[t_idx_train, label_col].values.astype(int)
    y_va = task_data.loc[t_idx_val,   label_col].values.astype(int)
    y_te = task_data.loc[t_idx_test,  label_col].values.astype(int)
    print(f"\n{'='*60}\nTask: {task}  pos_test={y_te.sum()} ({y_te.mean():.1%})")

    task_res = {"N_test": int(len(y_te)), "prevalence": float(y_te.mean())}

    for fname, fcols in FEATURE_SETS.items():
        fcols = [c for c in fcols if c in data.columns]
        if not fcols:
            continue

        X_tr = task_data.loc[t_idx_train, fcols].values.astype(float)
        X_va = task_data.loc[t_idx_val,   fcols].values.astype(float)
        X_te = task_data.loc[t_idx_test,  fcols].values.astype(float)

        col_means = np.nan_to_num(np.nanmean(X_tr, axis=0), nan=0.0)
        for arr in [X_tr, X_va, X_te]:
            nm = np.isnan(arr)
            if nm.any():
                arr[nm] = np.take(col_means, np.where(nm)[1])

        if y_tr.sum() < 10:
            continue

        model  = train_lgbm(X_tr, y_tr, X_va, y_va)
        y_prob = model.predict_proba(X_te)[:, 1]
        auc, lo, hi = bootstrap_auroc(y_te, y_prob)

        print(f"  {fname:<30}: AUROC={auc:.4f} [{lo:.4f}–{hi:.4f}]")
        task_res[fname] = {"auroc": round(auc, 4), "ci_lo": round(lo, 4),
                           "ci_hi": round(hi, 4), "n_features": len(fcols)}

    all_results[task] = task_res

# ── 7. Save results ──────────────────────────────────────────────────────────
with open(OUT / "temporal_trajectory_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

rows = []
for task, r in all_results.items():
    for fname, v in r.items():
        if isinstance(v, dict):
            rows.append({"task": task, "feature_set": fname, **v})
pd.DataFrame(rows).to_csv(OUT / "temporal_trajectory_auroc.csv", index=False)

# ── 8. Figure ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
})

MODEL_DISPLAY = {
    "flat_ordering_only":     ("Flat ordering\n(bin sums)",   "#56B4E9", None),
    "trajectory_only":        ("Trajectory\nfeatures only",   "#0072B2", "///"),
    "trajectory_enriched":    ("Trajectory\n+ bin counts",    "#0072B2", None),
    "static_plus_flat":       ("Static\n+ flat ordering",     "#009E73", None),
    "static_plus_trajectory": ("Static\n+ trajectory",        "#009E73", "///"),
}

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle(
    "Figure 18 — Temporal Ordering Trajectory: Does the Timing of Orders Matter?\n"
    "Flat ordering counts vs trajectory-enriched features (escalation slope, first-order timing, entropy)",
    fontsize=11, fontweight="bold", y=1.03
)

x = np.arange(len(MODEL_DISPLAY))
w = 0.55

for ax_idx, task in enumerate(TASKS):
    ax   = axes[ax_idx]
    r    = all_results.get(task, {})
    aurocs, lo_errs, hi_errs, colors, hatches = [], [], [], [], []

    for fname, (lbl, col, hatch) in MODEL_DISPLAY.items():
        v = r.get(fname, {})
        aurocs.append(v.get("auroc", 0))
        lo_errs.append(v.get("auroc", 0) - v.get("ci_lo", v.get("auroc", 0)))
        hi_errs.append(v.get("ci_hi", v.get("auroc", 0)) - v.get("auroc", 0))
        colors.append(col); hatches.append(hatch)

    bars = ax.bar(x, aurocs, width=w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for b, hatch in zip(bars, hatches):
        if hatch:
            b.set_hatch(hatch)
    ax.errorbar(x, aurocs, yerr=[lo_errs, hi_errs],
                fmt="none", color="#333333", capsize=4, linewidth=1.3)
    for xi, (a, he) in enumerate(zip(aurocs, hi_errs)):
        if a > 0:
            ax.text(xi, a + he + 0.003, f"{a:.3f}",
                    ha="center", fontsize=7.5, fontweight="bold", color="#333333")

    valid_aurocs = [a for a in aurocs if a > 0]
    if valid_aurocs:
        ymin = max(0.50, min(valid_aurocs) - 0.05)
        ymax = max(valid_aurocs) + max(hi_errs) + 0.06
        ax.set_ylim(ymin, min(ymax, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in MODEL_DISPLAY.values()],
                       fontsize=7.5, ha="center")
    ax.set_ylabel("AUROC" if ax_idx == 0 else "")
    ax.set_title(TASK_LABELS[task])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.text(0.98, 0.03,
            f"N={r.get('N_test', 0):,}\nPrev={r.get('prevalence', 0):.1%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")

import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=col, alpha=0.85, label=lbl)
           for _, (lbl, col, _) in MODEL_DISPLAY.items()]
fig.legend(handles=patches, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig(FIG / "fig18_temporal_trajectory.png", bbox_inches="tight")
plt.close()
print("\nSaved → fig18_temporal_trajectory.png")

print("\n" + "=" * 70)
print("TEMPORAL TRAJECTORY — SUMMARY")
print("=" * 70)
for task in TASKS:
    r = all_results.get(task, {})
    flat = r.get("flat_ordering_only", {}).get("auroc", 0)
    traj = r.get("trajectory_enriched", {}).get("auroc", 0)
    full = r.get("static_plus_trajectory", {}).get("auroc", 0)
    print(f"  {task:<20}: flat={flat:.4f}  trajectory={traj:.4f}"
          f"  Δ={traj-flat:+.4f}  full+traj={full:.4f}")
print("=" * 70)
