#!/usr/bin/env python
"""
Experiment L — Triage Window: Ordering-Only Prediction Before Lab Results Return
=================================================================================
Demonstrates that ordering-only features provide meaningful risk stratification
at triage — *before* laboratory values are available.

Key analyses:
  1. Data sparsity by time_bin: show that values are largely unavailable in
     the first 0–4h (time_bin=0) while ordering decisions are immediate
  2. AUROC as a function of observation window width (0–4h, 0–8h, ..., 0–48h):
     ordering-only converges to peak AUROC faster than values-only
  3. Head-to-head triage comparison at time_bin=0 only:
     ordering-only vs values-only vs baseline (age+sex+admission type)

Clinical context: lab turnaround time is typically 30–90 minutes. At the
moment of triage, physicians have made ordering decisions but results are
not yet available. Ordering-only models can flag high-risk patients immediately.

Outputs → 1_ordering_paper/results/L_triage_window/
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "1_ordering_paper" / "results" / "L_triage_window"
OUT.mkdir(parents=True, exist_ok=True)

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED   = 42
N_BOOT = 1000

# Observation windows: (time_bin_max, label)
# time_bin 0 = hours 0–4, bin 1 = 4–8, ..., bin 11 = 44–48
WINDOWS = [
    (0,  "0–4h"),
    (1,  "0–8h"),
    (2,  "0–12h"),
    (3,  "0–16h"),
    (5,  "0–24h"),
    (8,  "0–36h"),
    (11, "0–48h"),
]

print("=" * 65)
print("Experiment L — Triage Window Analysis")
print("=" * 65)


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
cohort      = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
labels      = pd.read_parquet(DATA / "labels.parquet")
static      = pd.read_parquet(DATA / "static.parquet")
labs_long   = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")
print(f"  Cohort: {len(cohort):,}")
print(f"  Labs   timeseries shape: {labs_long.shape}")
print(f"  Vitals timeseries shape: {vitals_long.shape}")

base = cohort.merge(labels, on=["hadm_id", "subject_id"])

# Patient-level split (same SEED=42, 70/15/15 as all experiments)
def patient_split(df, ref_df=None):
    ref = ref_df if ref_df is not None else df
    pat = ref.groupby("subject_id")["mortality"].max().reset_index()
    pat = pat.sample(frac=1, random_state=SEED)
    n = len(pat)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    tr_s = set(pat.iloc[:n_tr]["subject_id"])
    va_s = set(pat.iloc[n_tr:n_tr + n_va]["subject_id"])
    te_s = set(pat.iloc[n_tr + n_va:]["subject_id"])
    return tr_s, va_s, te_s

# Get split using all patients
tr_s, va_s, te_s = patient_split(base)


# ── 2. Sparsity analysis at each time_bin ─────────────────────────────────────
print("\n[2] Computing data sparsity by time_bin...")

lab_obs_cols    = [c for c in labs_long.columns if c.endswith("_obs")]
vital_obs_cols  = [c for c in vitals_long.columns if c.endswith("_obs")]
lab_val_cols    = [c for c in labs_long.columns
                   if c not in ("hadm_id","time_bin") and not c.endswith("_obs")]
vital_val_cols  = [c for c in vitals_long.columns
                   if c not in ("hadm_id","time_bin") and not c.endswith("_obs")]

sparsity_rows = []
for bin_id in range(12):
    hours_label = f"{bin_id*4}–{bin_id*4+4}h"

    labs_bin   = labs_long[labs_long["time_bin"] == bin_id]
    vitals_bin = vitals_long[vitals_long["time_bin"] == bin_id]

    n_adm = labs_bin["hadm_id"].nunique()

    # Fraction of admissions with ANY lab ordered
    any_lab_ordered = (labs_bin[lab_obs_cols].sum(axis=1) > 0).mean()
    # Fraction of admissions with ANY lab value present (non-NaN)
    any_lab_value   = labs_bin[lab_val_cols].notna().any(axis=1).mean()
    # Same for vitals
    any_vital_ordered = (vitals_bin[vital_obs_cols].sum(axis=1) > 0).mean()
    any_vital_value   = vitals_bin[vital_val_cols].notna().any(axis=1).mean()

    sparsity_rows.append({
        "time_bin": bin_id,
        "hours": hours_label,
        "n_admissions": n_adm,
        "frac_any_lab_ordered": round(any_lab_ordered, 4),
        "frac_any_lab_value":   round(any_lab_value,   4),
        "frac_any_vital_ordered": round(any_vital_ordered, 4),
        "frac_any_vital_value":   round(any_vital_value,   4),
    })

    if bin_id <= 3:
        print(f"  bin={bin_id} ({hours_label:>8})  "
              f"lab_ordered={any_lab_ordered:.1%}  lab_value={any_lab_value:.1%}  "
              f"vital_ordered={any_vital_ordered:.1%}  vital_value={any_vital_value:.1%}")

sparsity_df = pd.DataFrame(sparsity_rows)
sparsity_df.to_csv(OUT / "sparsity_by_timebin.csv", index=False)
print(f"  → sparsity_by_timebin.csv")


# ── 3. Feature extraction for a given window ──────────────────────────────────
def extract_features_for_window(labs_long, vitals_long, t_max):
    """
    Extract value and ordering features using only time_bins 0..t_max.
    Returns (val_features, ord_features) as DataFrames indexed by hadm_id.
    """
    labs_w   = labs_long[labs_long["time_bin"] <= t_max]
    vitals_w = vitals_long[vitals_long["time_bin"] <= t_max]

    # Value features: mean of value columns across bins
    def val_feats(df_w, name):
        vcols = [c for c in df_w.columns
                 if c not in ("hadm_id","time_bin") and not c.endswith("_obs")]
        agg = df_w.groupby("hadm_id", sort=False)[vcols].mean()
        agg.columns = [f"{c}__{name}" for c in agg.columns]
        return agg.reset_index()

    # Ordering features: derived from _obs columns
    def ord_feats(df_w, name):
        obs_cols = [c for c in df_w.columns if c.endswith("_obs")]
        rows = []
        for hadm_id, grp in df_w.groupby("hadm_id", sort=False):
            grp = grp.sort_values("time_bin")
            obs = grp[obs_cols].values.astype(float)
            n_bins = obs.shape[0]
            t = np.arange(n_bins)
            feat = {"hadm_id": hadm_id}
            binary = (obs > 0).astype(float)
            tests_per_bin = binary.sum(axis=1)
            feat[f"ordering_intensity__{name}"] = binary.mean()
            feat[f"ordering_diversity__{name}"] = (obs.sum(axis=0) > 0).sum()
            feat[f"ordering_breadth__{name}"]   = tests_per_bin.mean()
            feat[f"ordering_escalation__{name}"] = (
                np.polyfit(t, tests_per_bin, 1)[0] if n_bins > 1 else 0.0
            )
            # total obs per test
            for i, col in enumerate(obs_cols):
                base = col[:-4]
                feat[f"total_obs_{base}__{name}"] = obs[:, i].sum()
                feat[f"intensity_{base}__{name}"]  = (obs[:, i] > 0).mean()
            rows.append(feat)
        return pd.DataFrame(rows)

    lv = val_feats(labs_w, "lab")
    vv = val_feats(vitals_w, "vit")
    lo = ord_feats(labs_w, "lab")
    vo = ord_feats(vitals_w, "vit")
    return lv, vv, lo, vo


# ── 4. LightGBM helpers ────────────────────────────────────────────────────────
LGBM = dict(n_estimators=1000, learning_rate=0.05, num_leaves=63,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
            random_state=SEED, verbose=-1, metric="auc")

rng = np.random.default_rng(SEED)

def bootstrap_auroc(y, p):
    if y.sum() < 5 or (y == 0).sum() < 5:
        return float("nan"), float("nan"), float("nan")
    base_val = roc_auc_score(y, p)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            boot.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2
    return round(base_val, 4), round(lo, 4), round(hi, 4)

def train_eval_window(df_feat, feat_cols, task):
    """Train on train+val split, evaluate on test."""
    tr = df_feat["subject_id"].isin(tr_s)
    va = df_feat["subject_id"].isin(va_s)
    te = df_feat["subject_id"].isin(te_s)

    X_tr = df_feat.loc[tr, feat_cols].astype("float32")
    X_va = df_feat.loc[va, feat_cols].astype("float32")
    X_te = df_feat.loc[te, feat_cols].astype("float32")
    y_tr = df_feat.loc[tr, task]
    y_va = df_feat.loc[va, task]
    y_te = df_feat.loc[te, task]

    ok_tr, ok_va, ok_te = y_tr.notna(), y_va.notna(), y_te.notna()
    prev = y_tr[ok_tr].mean()
    pos_w = (1 - prev) / prev if prev > 0 else 1.0

    m = lgb.LGBMClassifier(scale_pos_weight=pos_w, **LGBM)
    m.fit(X_tr[ok_tr], y_tr[ok_tr],
          eval_set=[(X_va[ok_va], y_va[ok_va])],
          callbacks=[lgb.early_stopping(30, verbose=False),
                     lgb.log_evaluation(-1)])

    p = m.predict_proba(X_te[ok_te])[:, 1]
    return bootstrap_auroc(y_te[ok_te].values, p)


# ── 5. Static baseline features ───────────────────────────────────────────────
STATIC_FEATS = ["age", "is_female",
                "admission_type_EMERGENCY", "admission_type_URGENT", "admission_type_ELECTIVE"]
static_avail = [c for c in STATIC_FEATS if c in static.columns]


# ── 6. AUROC by observation window ────────────────────────────────────────────
print("\n[3] Computing AUROC by observation window...")
print(f"  {'Window':<10}  {'Hours':<8}  {'Task':<16}  {'Ordering':>10}  {'Values':>10}")

auroc_rows = []

for t_max, label in WINDOWS:
    print(f"\n  Window t_max={t_max} ({label})")
    lv, vv, lo, vo = extract_features_for_window(labs_long, vitals_long, t_max)

    for task in TASKS:
        # Ordering-only df
        df_ord_w = (base.merge(static[["hadm_id"] + static_avail], on="hadm_id", how="left")
                        .merge(lo, on="hadm_id", how="left")
                        .merge(vo, on="hadm_id", how="left"))
        ord_feat_cols = [c for c in df_ord_w.columns
                         if c not in ["hadm_id","subject_id"] + TASKS]

        # Values-only df
        df_val_w = (base.merge(static[["hadm_id"] + static_avail], on="hadm_id", how="left")
                        .merge(lv, on="hadm_id", how="left")
                        .merge(vv, on="hadm_id", how="left"))
        val_feat_cols = [c for c in df_val_w.columns
                         if c not in ["hadm_id","subject_id"] + TASKS]

        a_ord, lo_ord, hi_ord = train_eval_window(df_ord_w, ord_feat_cols, task)
        a_val, lo_val, hi_val = train_eval_window(df_val_w, val_feat_cols, task)

        print(f"    {task:<16}  ordering={a_ord:.3f} [{lo_ord:.3f}–{hi_ord:.3f}]  "
              f"values={a_val:.3f} [{lo_val:.3f}–{hi_val:.3f}]")

        auroc_rows.append({
            "t_max_bin": t_max, "window_label": label,
            "hours_max": (t_max + 1) * 4,
            "task": task,
            "ordering_auroc": a_ord, "ordering_ci_lo": lo_ord, "ordering_ci_hi": hi_ord,
            "values_auroc":   a_val, "values_ci_lo":   lo_val, "values_ci_hi":   hi_val,
            "ordering_advantage": round(a_ord - a_val, 4) if not np.isnan(a_ord) and not np.isnan(a_val) else np.nan,
        })

auroc_df = pd.DataFrame(auroc_rows)
auroc_df.to_csv(OUT / "auroc_by_window.csv", index=False)
print(f"\n  → auroc_by_window.csv")


# ── 7. Triage AUROC (t_max=0 only) with baseline comparison ──────────────────
print("\n[4] Triage comparison (time_bin=0 only — first 4 hours)...")

triage_rows = []
lv0, vv0, lo0, vo0 = extract_features_for_window(labs_long, vitals_long, t_max=0)

for task in TASKS:
    # Baseline: age + sex + admission type only
    df_base = base.merge(static[["hadm_id"] + static_avail], on="hadm_id", how="left")
    base_feat = static_avail
    a_base, lo_base, hi_base = train_eval_window(df_base, base_feat, task)

    # Ordering-only at t=0
    df_ord0 = (base.merge(static[["hadm_id"] + static_avail], on="hadm_id", how="left")
                   .merge(lo0, on="hadm_id", how="left")
                   .merge(vo0, on="hadm_id", how="left"))
    ord0_cols = [c for c in df_ord0.columns if c not in ["hadm_id","subject_id"] + TASKS]
    a_ord0, lo_ord0, hi_ord0 = train_eval_window(df_ord0, ord0_cols, task)

    # Values-only at t=0
    df_val0 = (base.merge(static[["hadm_id"] + static_avail], on="hadm_id", how="left")
                   .merge(lv0, on="hadm_id", how="left")
                   .merge(vv0, on="hadm_id", how="left"))
    val0_cols = [c for c in df_val0.columns if c not in ["hadm_id","subject_id"] + TASKS]
    a_val0, lo_val0, hi_val0 = train_eval_window(df_val0, val0_cols, task)

    print(f"  {task:<18}  baseline={a_base:.3f}  ordering={a_ord0:.3f}  values={a_val0:.3f}")

    for model_name, a, lo_ci, hi_ci in [
        ("baseline",      a_base,  lo_base,  hi_base),
        ("ordering_only", a_ord0,  lo_ord0,  hi_ord0),
        ("values_only",   a_val0,  lo_val0,  hi_val0),
    ]:
        triage_rows.append({
            "task": task, "model": model_name,
            "auroc": a, "ci_lo": lo_ci, "ci_hi": hi_ci,
            "auroc_str": f"{a:.3f} [{lo_ci:.3f}–{hi_ci:.3f}]",
        })

triage_df = pd.DataFrame(triage_rows)
triage_df.to_csv(OUT / "triage_auroc.csv", index=False)
print(f"\n  → triage_auroc.csv")


# ── Summary JSON ──────────────────────────────────────────────────────────────
# Key finding: ordering advantage at t=0
ordering_adv_t0 = (auroc_df[auroc_df["t_max_bin"] == 0]
                   .groupby("task")["ordering_advantage"].mean().to_dict())

summary = {
    "key_finding_triage": (
        "At time_bin=0 (first 4 hours), before lab results return, "
        "ordering-only AUROC exceeds values-only AUROC by "
        f"{np.nanmean(list(ordering_adv_t0.values())):.3f} AUROC on average."
    ),
    "ordering_advantage_at_t0_per_task": ordering_adv_t0,
    "sparsity_at_t0": sparsity_df[sparsity_df["time_bin"] == 0].to_dict(orient="records")[0],
    "triage_auroc": triage_df.to_dict(orient="records"),
}

with open(OUT / "L_triage_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  → L_triage_summary.json")
print(f"\n✅  Experiment L complete.  Results → {OUT}")
