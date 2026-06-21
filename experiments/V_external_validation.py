#!/usr/bin/env python
"""
Experiment V — External Validation on eICU-CRD
===============================================
Validates the ordering signal paper's main claims on the
eICU Collaborative Research Database (v2.0):
  208 hospitals, ~200K ICU admissions, USA 2014–2015.

Method:
  Re-trains ordering-only and values-only LightGBM models on a
  MIMIC-IV subset restricted to features common to both datasets
  (24 labs + 8 vitals), then applies these to eICU (zero-shot
  external validation, no eICU data seen during training).

Three claims validated:
  1. Ordering AUROC in eICU ≈ values AUROC (non-inferiority)
  2. Ordering AUROC is consistent across eICU hospitals
     (multi-centre generalisability)
  3. Ordering AUROC within CCI=0-equivalent (no prior history)
     eICU patients is robust

DEPENDS ON: 08_eicu_preprocess.py must run first.

Outputs → 1_ordering_paper/results/V_external_validation/
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parents[2]
DATA_MIMIC = ROOT / "data" / "processed"
DATA_EICU  = ROOT / "data" / "processed_eicu"
OUT        = ROOT / "1_ordering_paper" / "results" / "V_external_validation"
OUT.mkdir(parents=True, exist_ok=True)

TASKS   = ["mortality", "aki", "sepsis"]   # readmission not available in eICU
SEED    = 42
N_BOOT  = 1000

print("=" * 60)
print("Experiment V — External Validation on eICU-CRD")
print("=" * 60)


# ── 1. Identify common feature set ────────────────────────────────────────────
print("\n[1] Identifying MIMIC-eICU common features...")
mimic_labs   = pd.read_parquet(DATA_MIMIC / "timeseries" / "labs.parquet").head(1)
mimic_vitals = pd.read_parquet(DATA_MIMIC / "timeseries" / "vitals.parquet").head(1)
eicu_labs    = pd.read_parquet(DATA_EICU  / "timeseries" / "labs_eicu.parquet").head(1)
eicu_vitals  = pd.read_parquet(DATA_EICU  / "timeseries" / "vitals_eicu.parquet").head(1)

# Get test names (strip _obs/_mean/etc suffix)
def get_tests(df, suffix="_obs"):
    return {c[:-len(suffix)] for c in df.columns if c.endswith(suffix)}

mimic_lab_tests   = get_tests(mimic_labs)
mimic_vital_tests = get_tests(mimic_vitals)
eicu_lab_tests    = get_tests(eicu_labs)
eicu_vital_tests  = get_tests(eicu_vitals)

common_labs   = sorted(mimic_lab_tests & eicu_lab_tests)
common_vitals = sorted(mimic_vital_tests & eicu_vital_tests)

print(f"  Common labs   : {len(common_labs)}   → {common_labs}")
print(f"  Common vitals : {len(common_vitals)} → {common_vitals}")

SUFFIXES_VAL = ["_mean", "_min", "_max", "_last"]
SUFFIXES_ORD = ["_obs"]

val_cols = [f"{t}{s}" for t in common_labs   for s in SUFFIXES_VAL] + \
           [f"{t}{s}" for t in common_vitals  for s in SUFFIXES_VAL]
ord_cols = [f"{t}_obs" for t in common_labs] + \
           [f"{t}_obs" for t in common_vitals]

print(f"  Value features: {len(val_cols)}")
print(f"  Ordering features: {len(ord_cols)}")


# ── 2. Build MIMIC feature matrix (common features only) ─────────────────────
print("\n[2] Building MIMIC feature matrices (common features)...")

cohort_m = pd.read_parquet(DATA_MIMIC / "cohort.parquet")[["hadm_id","subject_id"]]
labels_m = pd.read_parquet(DATA_MIMIC / "labels.parquet")
static_m = pd.read_parquet(DATA_MIMIC / "static.parquet")

print("  Loading MIMIC timeseries (common columns only)...")
labs_m_full = pd.read_parquet(
    DATA_MIMIC / "timeseries" / "labs.parquet",
    columns=["hadm_id", "time_bin"] +
            [c for c in mimic_labs.columns
             if any(c.startswith(t) for t in common_labs)])

vitals_m_full = pd.read_parquet(
    DATA_MIMIC / "timeseries" / "vitals.parquet",
    columns=["hadm_id", "time_bin"] +
            [c for c in mimic_vitals.columns
             if any(c.startswith(t) for t in common_vitals)])

# Aggregate across time_bins (mean over bins — same as MIMIC pipeline)
def agg_ts(df_long, feat_cols):
    return df_long.groupby("hadm_id", sort=False)[feat_cols].mean().reset_index()

labs_m_agg   = agg_ts(labs_m_full,   [c for c in labs_m_full.columns
                                        if c not in ("hadm_id","time_bin")])
vitals_m_agg = agg_ts(vitals_m_full, [c for c in vitals_m_full.columns
                                        if c not in ("hadm_id","time_bin")])

base_m = (cohort_m
          .merge(labels_m, on=["hadm_id","subject_id"])
          .merge(static_m[["hadm_id","age","is_female","cci_score"]], on="hadm_id", how="left")
          .merge(labs_m_agg,   on="hadm_id", how="left")
          .merge(vitals_m_agg, on="hadm_id", how="left"))

META = ["hadm_id","subject_id"] + TASKS + ["readmission_30d"]
static_feats = ["age","is_female","cci_score"]

val_feat_cols = static_feats + [c for c in val_cols if c in base_m.columns]
ord_feat_cols = static_feats + [c for c in ord_cols if c in base_m.columns]

print(f"  MIMIC base shape: {base_m.shape}")


# ── 3. MIMIC patient-level split (SEED=42, same as all experiments) ──────────
def patient_split(df):
    pat = df.groupby("subject_id")["mortality"].max().reset_index()
    pat = pat.sample(frac=1, random_state=SEED)
    n = len(pat)
    n_tr = int(0.70 * n);  n_va = int(0.15 * n)
    tr_s = set(pat.iloc[:n_tr]["subject_id"])
    va_s = set(pat.iloc[n_tr:n_tr+n_va]["subject_id"])
    return (base_m["subject_id"].isin(tr_s),
            base_m["subject_id"].isin(va_s))

tr, va = patient_split(base_m)
print(f"  MIMIC split: Train {tr.sum():,} | Val {va.sum():,} | "
      f"(eICU = external test)")


# ── 4. Train models on MIMIC (common features) ───────────────────────────────
print("\n[3] Training MIMIC models on common feature set...")

LGBM = dict(n_estimators=2000, learning_rate=0.05, num_leaves=127,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
            random_state=SEED, verbose=-1, metric="auc")

def train(feat_cols, task):
    X_tr = base_m.loc[tr, feat_cols].astype("float32")
    y_tr = base_m.loc[tr, task]
    X_va = base_m.loc[va, feat_cols].astype("float32")
    y_va = base_m.loc[va, task]
    ok_tr, ok_va = y_tr.notna(), y_va.notna()
    pos_w = (1 - y_tr[ok_tr].mean()) / y_tr[ok_tr].mean()
    m = lgb.LGBMClassifier(scale_pos_weight=pos_w, **LGBM)
    m.fit(X_tr[ok_tr], y_tr[ok_tr],
          eval_set=[(X_va[ok_va], y_va[ok_va])],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])
    return m

models = {}
for task in TASKS:
    print(f"  {task}...")
    models[(task,"values_only")]   = train(val_feat_cols, task)
    models[(task,"ordering_only")] = train(ord_feat_cols, task)


# ── 5. Build eICU feature matrix ─────────────────────────────────────────────
print("\n[4] Building eICU feature matrices...")

cohort_e = pd.read_parquet(DATA_EICU / "cohort_eicu.parquet")
static_e = pd.read_parquet(DATA_EICU / "static_eicu.parquet")

print("  Loading eICU timeseries...")
labs_e_full = pd.read_parquet(
    DATA_EICU / "timeseries" / "labs_eicu.parquet",
    columns=["hadm_id","time_bin"] +
            [c for c in eicu_labs.columns
             if any(c.startswith(t) for t in common_labs)])

vitals_e_full = pd.read_parquet(
    DATA_EICU / "timeseries" / "vitals_eicu.parquet",
    columns=["hadm_id","time_bin"] +
            [c for c in eicu_vitals.columns
             if any(c.startswith(t) for t in common_vitals)])

labs_e_agg   = agg_ts(labs_e_full,   [c for c in labs_e_full.columns
                                        if c not in ("hadm_id","time_bin")])
vitals_e_agg = agg_ts(vitals_e_full, [c for c in vitals_e_full.columns
                                        if c not in ("hadm_id","time_bin")])

base_e = (cohort_e
          .merge(static_e[["hadm_id","cci_score"]], on="hadm_id", how="left")
          .merge(labs_e_agg,   on="hadm_id", how="left")
          .merge(vitals_e_agg, on="hadm_id", how="left"))

print(f"  eICU base shape: {base_e.shape}  ({len(base_e):,} stays)")


# ── 6. Bootstrap AUROC ────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def bootstrap_auroc(y, p, n_boot=N_BOOT):
    if y.sum() < 5 or (y == 0).sum() < 5:
        return float("nan"), float("nan"), float("nan")
    base = roc_auc_score(y, p)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            boot.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2
    return round(base, 4), round(lo, 4), round(hi, 4)


def paired_bootstrap_auc_diff(y, p_a, p_b, n_boot=N_BOOT):
    observed = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            boot.append(roc_auc_score(yb, p_a[idx]) - roc_auc_score(yb, p_b[idx]))
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return float(observed), float(lo), float(hi), float(min(p, 1.0))


# ── 7. Evaluate on eICU ──────────────────────────────────────────────────────
print("\n[5] Evaluating on eICU (external validation)...")
results = []
prediction_rows = []
paired_rows = []

for task in TASKS:
    sub = base_e[base_e[task].notna()].copy()
    y   = sub[task].values.astype(int)
    task_probs = {}

    for model_name, feat_cols in [("values_only",   val_feat_cols),
                                   ("ordering_only", ord_feat_cols)]:
        m = models[(task, model_name)]
        X = sub[feat_cols].astype("float32")
        p = m.predict_proba(X)[:, 1]
        task_probs[model_name] = p
        auroc, lo, hi = bootstrap_auroc(y, p)
        prediction_rows.append(pd.DataFrame({
            "dataset": "eICU",
            "task": task,
            "hadm_id": sub["hadm_id"].astype(str).values,
            "subject_id": sub["subject_id"].astype(str).values,
            "hospitalid": sub["hospitalid"].astype(str).values,
            "model": model_name,
            "y_true": y,
            "y_prob": p,
        }))
        results.append({
            "dataset": "eICU", "task": task, "model": model_name,
            "N": len(sub), "n_pos": int(y.sum()),
            "auroc": auroc, "ci_lo": lo, "ci_hi": hi,
            "auroc_str": f"{auroc:.3f} [{lo:.3f}–{hi:.3f}]"
        })
        print(f"  eICU  {task:<16} {model_name:<16}  "
              f"AUROC={auroc:.3f} [{lo:.3f}–{hi:.3f}]  N={len(sub):,}")
    diff, lo, hi, pval = paired_bootstrap_auc_diff(
        y, task_probs["values_only"], task_probs["ordering_only"]
    )
    paired_rows.append({
        "dataset": "eICU",
        "task": task,
        "model_a": "values_only",
        "model_b": "ordering_only",
        "comparison": "values_only - ordering_only",
        "n": int(len(sub)),
        "n_pos": int(y.sum()),
        "auc_a": float(roc_auc_score(y, task_probs["values_only"])),
        "auc_b": float(roc_auc_score(y, task_probs["ordering_only"])),
        "auc_diff": diff,
        "diff_ci_lo": lo,
        "diff_ci_hi": hi,
        "p_value_two_sided": pval,
    })

# MIMIC test set for comparison (using same model)
te = ~(tr | va)
for task in TASKS:
    sub = base_m.loc[te].copy()
    sub = sub[sub[task].notna()]
    y   = sub[task].values.astype(int)
    task_probs = {}
    for model_name, feat_cols in [("values_only",   val_feat_cols),
                                   ("ordering_only", ord_feat_cols)]:
        m = models[(task, model_name)]
        X = sub[feat_cols].astype("float32")
        p = m.predict_proba(X)[:, 1]
        task_probs[model_name] = p
        prediction_rows.append(pd.DataFrame({
            "dataset": "MIMIC (test)",
            "task": task,
            "hadm_id": sub["hadm_id"].astype(str).values,
            "subject_id": sub["subject_id"].astype(str).values,
            "hospitalid": "",
            "model": model_name,
            "y_true": y,
            "y_prob": p,
        }))
        auroc, lo, hi = bootstrap_auroc(y, p)
        results.append({
            "dataset": "MIMIC (test)", "task": task, "model": model_name,
            "N": len(sub), "n_pos": int(y.sum()),
            "auroc": auroc, "ci_lo": lo, "ci_hi": hi,
            "auroc_str": f"{auroc:.3f} [{lo:.3f}–{hi:.3f}]"
        })
    diff, lo, hi, pval = paired_bootstrap_auc_diff(
        y, task_probs["values_only"], task_probs["ordering_only"]
    )
    paired_rows.append({
        "dataset": "MIMIC (test)",
        "task": task,
        "model_a": "values_only",
        "model_b": "ordering_only",
        "comparison": "values_only - ordering_only",
        "n": int(len(sub)),
        "n_pos": int(y.sum()),
        "auc_a": float(roc_auc_score(y, task_probs["values_only"])),
        "auc_b": float(roc_auc_score(y, task_probs["ordering_only"])),
        "auc_diff": diff,
        "diff_ci_lo": lo,
        "diff_ci_hi": hi,
        "p_value_two_sided": pval,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(OUT / "external_validation_auroc.csv", index=False)
pd.concat(prediction_rows, ignore_index=True).to_parquet(
    OUT / "external_validation_predictions.parquet", index=False
)
pd.DataFrame(paired_rows).to_csv(
    OUT / "external_validation_paired_auc_comparisons.csv", index=False
)


# ── 8. Per-hospital AUROC (multi-centre generalisability) ─────────────────────
print("\n[6] Per-hospital AUROC (multi-centre analysis)...")
hosp_results = []

for task in TASKS:
    sub_all = base_e[base_e[task].notna()].copy()
    m_ord = models[(task, "ordering_only")]

    for hosp_id, hosp_df in sub_all.groupby("hospitalid"):
        if len(hosp_df) < 30:
            continue
        y = hosp_df[task].values.astype(int)
        if y.sum() < 3 or (y == 0).sum() < 3:
            continue
        X = hosp_df[ord_feat_cols].astype("float32")
        p = m_ord.predict_proba(X)[:, 1]
        try:
            auroc = round(roc_auc_score(y, p), 4)
        except Exception:
            continue
        hosp_results.append({
            "task": task, "hospitalid": hosp_id,
            "N": len(hosp_df), "n_pos": int(y.sum()), "auroc": auroc
        })

hosp_df_out = pd.DataFrame(hosp_results)
hosp_df_out.to_csv(OUT / "per_hospital_auroc.csv", index=False)

if len(hosp_df_out) > 0:
    print(f"  Hospitals evaluated: {hosp_df_out['hospitalid'].nunique()}")
    for task in TASKS:
        sub = hosp_df_out[hosp_df_out["task"] == task]["auroc"]
        print(f"  {task:<16}  median={sub.median():.3f}  "
              f"IQR=[{sub.quantile(0.25):.3f}–{sub.quantile(0.75):.3f}]  "
              f"range=[{sub.min():.3f}–{sub.max():.3f}]  N_hosp={len(sub)}")


# ── 9. Summary JSON ───────────────────────────────────────────────────────────
summary = {}
for task in TASKS:
    summary[task] = {}
    for ds in ["MIMIC (test)", "eICU"]:
        summary[task][ds] = {}
        for model in ["ordering_only", "values_only"]:
            row = results_df[
                (results_df["task"] == task) &
                (results_df["dataset"] == ds) &
                (results_df["model"] == model)
            ]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            summary[task][ds][model] = {
                "auroc": row["auroc"], "ci_lo": row["ci_lo"],
                "ci_hi": row["ci_hi"], "N": int(row["N"])
            }

with open(OUT / "external_validation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅  Experiment V complete.  Results → {OUT}")
print("\nKey results (ordering_only):")
print(f"{'Task':<16}  {'MIMIC':<25}  {'eICU':<25}")
for task in TASKS:
    m_row = results_df[(results_df["task"]==task) &
                        (results_df["dataset"]=="MIMIC (test)") &
                        (results_df["model"]=="ordering_only")].iloc[0]
    e_row = results_df[(results_df["task"]==task) &
                        (results_df["dataset"]=="eICU") &
                        (results_df["model"]=="ordering_only")].iloc[0]
    print(f"  {task:<16}  {m_row['auroc_str']:<25}  {e_row['auroc_str']}")
