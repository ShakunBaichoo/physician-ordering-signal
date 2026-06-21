#!/usr/bin/env python
"""
Experiment B — Conformal Prediction Equity Analysis
====================================================
Tests the hypothesis that model uncertainty (conformal prediction coverage)
is systematically miscalibrated for:
  1. Underrepresented demographic groups (race, insurance, gender)
  2. Patients where physicians are most uncertain (high ordering intensity)

Uses the combined LightGBM models from Experiment A as the base predictor.

Method: Split Conformal Prediction (Angelopoulos & Bates 2022)
  - Calibrate on val set → threshold τ at target coverage (90%)
  - Evaluate coverage on test set, stratified by subgroup

Extension: Mondrian Conformal Prediction
  - Separate τ per subgroup → restores conditional coverage
  - Shows that ordering-stratified calibration closes the equity gap

Outputs (results/novel/B_conformal_equity/):
  - coverage_by_subgroup.csv    marginal vs conditional coverage
  - mondrian_coverage.csv       coverage after stratified calibration
  - calibration_summary.json    full metrics
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
SRC  = ROOT / "1_ordering_paper" / "results" / "A_ordering_signal"
OUT  = ROOT / "1_ordering_paper" / "results" / "B_conformal_equity"
OUT.mkdir(parents=True, exist_ok=True)

TASKS   = ["mortality", "readmission_30d", "aki", "sepsis"]
ALPHA   = 0.10   # target miscoverage → 90% coverage
SEED    = 42


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
cohort  = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id", "race", "insurance"]]
labels  = pd.read_parquet(DATA / "labels.parquet")
static  = pd.read_parquet(DATA / "static.parquet")

# Ordering intensity (computed in Experiment A)
ord_path = SRC / "ordering_features.csv"
if not ord_path.exists():
    raise FileNotFoundError("Run A_ordering_signal.py first to generate ordering_features.csv")
ordering = pd.read_csv(ord_path)

print("  Loading labs & vitals for combined features...")
labs_long   = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

def agg_values(df_long):
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id", "time_bin") and not c.endswith("_obs")]
    return df_long.groupby("hadm_id", sort=False)[val_cols].mean().reset_index()

def agg_ordering(df_long, name):
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[obs_cols].sum()
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()

labs_val   = agg_values(labs_long)
vitals_val = agg_values(vitals_long)
labs_ord   = agg_ordering(labs_long, "lab")
vitals_ord = agg_ordering(vitals_long, "vit")

df = (cohort
      .merge(labels,     on=["hadm_id", "subject_id"])
      .merge(static,     on="hadm_id", how="left")
      .merge(labs_val,   on="hadm_id", how="left")
      .merge(vitals_val, on="hadm_id", how="left")
      .merge(labs_ord,   on="hadm_id", how="left")
      .merge(vitals_ord, on="hadm_id", how="left")
      .merge(ordering,   on="hadm_id", how="left"))

meta_cols = ["hadm_id", "subject_id", "race", "insurance"] + TASKS
feat_cols = [c for c in df.columns if c not in meta_cols]


# ── 2. Patient-level split (same seed as Exp A) ───────────────────────────────
pat = df.groupby("subject_id")["mortality"].max().reset_index()
pat = pat.sample(frac=1, random_state=SEED)
n = len(pat)
n_train, n_val = int(0.70 * n), int(0.15 * n)
train_s = set(pat.iloc[:n_train]["subject_id"])
val_s   = set(pat.iloc[n_train:n_train + n_val]["subject_id"])
test_s  = set(pat.iloc[n_train + n_val:]["subject_id"])

tr_m = df["subject_id"].isin(train_s)
va_m = df["subject_id"].isin(val_s)
te_m = df["subject_id"].isin(test_s)


# ── 3. Conformal prediction utilities ────────────────────────────────────────

def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal threshold: smallest τ such that ≥ (1-α) of
    calibration nonconformity scores are ≤ τ.
    Nonconformity score for binary classification: 1 - p(true label).
    """
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    return np.quantile(scores, level)


def nonconformity_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1 - predicted probability of the true class."""
    return 1 - np.where(labels == 1, probs, 1 - probs)


def prediction_set_size(probs: np.ndarray, tau: float) -> np.ndarray:
    """
    For binary classification the prediction set is:
      {1}      if p(1) ≥ 1-τ
      {0}      if p(0) ≥ 1-τ  i.e. p(1) ≤ τ
      {0,1}    otherwise (uncertain)
    Returns set size: 1 or 2.
    """
    in_1 = (probs >= 1 - tau)
    in_0 = (probs <= tau)
    return 1 + (in_1 & in_0).astype(int)  # 2 if both in set, 1 otherwise


def coverage(probs: np.ndarray, labels: np.ndarray, tau: float) -> float:
    """Fraction of test points whose true label is in the prediction set."""
    covered = np.where(labels == 1, probs >= 1 - tau, probs <= tau)
    return covered.mean()


# ── 4. Train combined model + run conformal analysis ─────────────────────────
all_results = {}

for task in TASKS:
    print(f"\n{'='*60}")
    print(f"  Task: {task}")
    print(f"{'='*60}")

    y_tr = df.loc[tr_m, task]; y_va = df.loc[va_m, task]; y_te = df.loc[te_m, task]
    ok_tr, ok_va, ok_te = y_tr.notna(), y_va.notna(), y_te.notna()

    X_tr = df.loc[tr_m, feat_cols].astype("float32")
    X_va = df.loc[va_m, feat_cols].astype("float32")
    X_te = df.loc[te_m, feat_cols].astype("float32")

    print(f"  Training combined model...")
    prev  = y_tr[ok_tr].mean()
    pos_w = (1 - prev) / prev
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pos_w,
        n_jobs=-1, random_state=SEED, verbose=-1, metric="auc",
    )
    model.fit(X_tr[ok_tr], y_tr[ok_tr],
              eval_set=[(X_va[ok_va], y_va[ok_va])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    val_probs  = model.predict_proba(X_va[ok_va])[:, 1]
    test_probs = model.predict_proba(X_te[ok_te])[:, 1]

    val_labels  = y_va[ok_va].values
    test_labels = y_te[ok_te].values
    test_df     = df.loc[te_m][ok_te].copy()
    test_df["prob"] = test_probs

    auroc = roc_auc_score(test_labels, test_probs)
    auprc = average_precision_score(test_labels, test_probs)
    print(f"  Test AUROC {auroc:.4f} | AUPRC {auprc:.4f}")

    # ── Split conformal calibration on val set ────────────────────────────────
    cal_scores = nonconformity_scores(val_probs, val_labels)
    tau = conformal_threshold(cal_scores, ALPHA)
    marginal_cov = coverage(test_probs, test_labels, tau)
    mean_set_size = prediction_set_size(test_probs, tau).mean()
    print(f"  Conformal τ={tau:.4f} | Marginal coverage={marginal_cov:.4f} "
          f"(target={1-ALPHA:.2f}) | Mean set size={mean_set_size:.3f}")

    # ── Subgroup coverage analysis ────────────────────────────────────────────
    subgroup_results = []

    def subgroup_coverage(mask, name, group_val):
        if mask.sum() < 30:
            return
        sub_probs  = test_probs[mask]
        sub_labels = test_labels[mask]
        cov = coverage(sub_probs, sub_labels, tau)
        set_sz = prediction_set_size(sub_probs, tau).mean()
        # Mondrian threshold: calibrate τ using val set patients in same subgroup
        # (need val-set subgroup membership)
        subgroup_results.append({
            "subgroup": name,
            "group": str(group_val),
            "n": int(mask.sum()),
            "coverage": round(float(cov), 4),
            "mean_set_size": round(float(set_sz), 4),
            "coverage_gap": round(float(cov - (1 - ALPHA)), 4),
        })

    # Race
    for race in test_df["race"].dropna().unique():
        mask = (test_df["race"] == race).values
        subgroup_coverage(mask, "race", race)

    # Insurance
    for ins in test_df["insurance"].dropna().unique():
        mask = (test_df["insurance"] == ins).values
        subgroup_coverage(mask, "insurance", ins)

    # Gender
    for g_val, g_name in [(1, "female"), (0, "male")]:
        mask = (test_df["is_female"] == g_val).values
        subgroup_coverage(mask, "gender", g_name)

    # Age quartile
    test_df["age_q"] = pd.qcut(test_df["age"], 4, labels=["Q1","Q2","Q3","Q4"])
    for q in ["Q1","Q2","Q3","Q4"]:
        mask = (test_df["age_q"] == q).values
        subgroup_coverage(mask, "age_quartile", q)

    # Ordering intensity quartile (key novel analysis)
    oi_cols = [c for c in test_df.columns if "ordering_intensity" in c]
    if oi_cols:
        test_df["ordering_intensity"] = test_df[oi_cols].mean(axis=1)
        test_df["ord_q"] = pd.qcut(test_df["ordering_intensity"], 4,
                                    labels=["Q1_low","Q2","Q3","Q4_high"],
                                    duplicates="drop")
        for q in test_df["ord_q"].dropna().unique():
            mask = (test_df["ord_q"] == q).values
            subgroup_coverage(mask, "ordering_intensity_quartile", q)

    sg_df = pd.DataFrame(subgroup_results)
    sg_df.to_csv(OUT / f"coverage_subgroups_{task}.csv", index=False)

    # ── Mondrian conformal: per-subgroup calibration ──────────────────────────
    # Re-calibrate τ separately for ordering intensity quartiles
    mondrian_results = []
    if oi_cols:
        val_df = df.loc[va_m][ok_va].copy()
        val_df["prob"] = val_probs
        val_df["ordering_intensity"] = val_df[oi_cols].mean(axis=1)
        val_df["ord_q"] = pd.qcut(val_df["ordering_intensity"], 4,
                                   labels=["Q1_low","Q2","Q3","Q4_high"],
                                   duplicates="drop")

        for q in test_df["ord_q"].dropna().unique():
            val_sub_mask = (val_df["ord_q"] == q).values
            if val_sub_mask.sum() < 30:
                continue
            cal_sub = nonconformity_scores(val_probs[val_sub_mask],
                                           val_labels[val_sub_mask])
            tau_q = conformal_threshold(cal_sub, ALPHA)

            test_sub_mask = (test_df["ord_q"] == q).values
            if test_sub_mask.sum() < 30:
                continue
            cov_q = coverage(test_probs[test_sub_mask],
                             test_labels[test_sub_mask], tau_q)
            mondrian_results.append({
                "ordering_quartile": str(q),
                "n_cal": int(val_sub_mask.sum()),
                "n_test": int(test_sub_mask.sum()),
                "tau_mondrian": round(float(tau_q), 4),
                "tau_global": round(float(tau), 4),
                "coverage_global_tau": round(float(
                    coverage(test_probs[test_sub_mask],
                             test_labels[test_sub_mask], tau)), 4),
                "coverage_mondrian_tau": round(float(cov_q), 4),
            })

        mondrian_df = pd.DataFrame(mondrian_results)
        mondrian_df.to_csv(OUT / f"mondrian_coverage_{task}.csv", index=False)

        print(f"\n  Mondrian conformal (ordering-stratified):")
        print(f"  {'Quartile':<12} {'Global τ cov':>14} {'Mondrian cov':>14} {'N':>6}")
        for _, r in mondrian_df.iterrows():
            print(f"  {r['ordering_quartile']:<12} "
                  f"{r['coverage_global_tau']:>14.4f} "
                  f"{r['coverage_mondrian_tau']:>14.4f} "
                  f"{r['n_test']:>6,}")

    # Print subgroup coverage summary
    print(f"\n  Subgroup coverage (target={1-ALPHA:.2f}):")
    print(f"  {'Subgroup':<30} {'Group':<25} {'N':>6} {'Coverage':>10} {'Gap':>8}")
    print(f"  {'-'*80}")
    for _, r in sg_df.sort_values("coverage_gap").iterrows():
        flag = " ⚠" if abs(r["coverage_gap"]) > 0.02 else ""
        print(f"  {r['subgroup']:<30} {str(r['group']):<25} "
              f"{r['n']:>6,} {r['coverage']:>10.4f} {r['coverage_gap']:>+8.4f}{flag}")

    all_results[task] = {
        "auroc": auroc, "auprc": auprc,
        "conformal_tau": float(tau),
        "marginal_coverage": float(marginal_cov),
        "mean_set_size": float(mean_set_size),
        "subgroup_n": len(subgroup_results),
    }

with open(OUT / "calibration_summary.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nAll results saved to {OUT}/")
print("Key files:")
print("  coverage_subgroups_<task>.csv  — marginal vs conditional coverage by subgroup")
print("  mondrian_coverage_<task>.csv   — ordering-stratified Mondrian conformal")
print("  calibration_summary.json       — overall metrics")
