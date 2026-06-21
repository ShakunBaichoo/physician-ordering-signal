#!/usr/bin/env python
"""
Experiment D — Shapley Ordering Attribution
============================================
Uses SHAP (cooperative game theory) to formally decompose each prediction
into contributions from three "players":
  - static      : demographics, comorbidities, medications
  - values      : lab/vital measurement values
  - ordering    : physician ordering frequencies (_obs columns)

Key question: For "deceptively normal" patients (normal values, high ordering
intensity), does the ordering player dominate the values player?

If yes → formal game-theoretic proof that test-ordering patterns are the
primary driver of risk prediction for this patient subgroup.

Method:
  - TreeExplainer (exact Shapley values for tree models, O(TLD^2))
  - Aggregate |SHAP| by feature group per patient
  - Compare group attributions across patient phenotypes

Outputs (results/novel/D_shapley_ordering/):
  - shap_group_attributions.csv    per-patient group |SHAP| sums
  - phenotype_attribution.csv      mean group attribution by phenotype
  - shap_summary_<task>.csv        top features overall
"""

import json
import warnings
from pathlib import Path
from itertools import combinations

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
SRC_A = ROOT / "1_ordering_paper" / "results" / "A_ordering_signal"
OUT   = ROOT / "1_ordering_paper" / "results" / "D_shapley_ordering"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED  = 42


# ── 1. Rebuild the combined feature matrix (same as Exp A) ───────────────────
print("Loading data...")
cohort  = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
labels  = pd.read_parquet(DATA / "labels.parquet")
static  = pd.read_parquet(DATA / "static.parquet")

print("  Labs...")
labs_long = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
print("  Vitals...")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

def agg_values(df_long, name):
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id", "time_bin") and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()

def agg_ordering(df_long, name):
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    total = df_long.groupby("hadm_id", sort=False)[obs_cols].agg(
        ["sum", "mean", lambda x: (x > 0).mean()])
    total.columns = [f"{col}_{stat}__{name}"
                     for col, stat in total.columns]
    # also add admission-level ordering intensity
    binary  = df_long.groupby("hadm_id", sort=False)[obs_cols].apply(
        lambda g: (g > 0).values.mean())
    total[f"ordering_intensity__{name}"] = binary.values
    return total.reset_index()

labs_val   = agg_values(labs_long,   "lab")
vitals_val = agg_values(vitals_long, "vit")
labs_ord   = agg_ordering(labs_long,   "lab")
vitals_ord = agg_ordering(vitals_long, "vit")

df = (cohort
      .merge(labels,     on=["hadm_id", "subject_id"])
      .merge(static,     on="hadm_id",  how="left")
      .merge(labs_val,   on="hadm_id",  how="left")
      .merge(vitals_val, on="hadm_id",  how="left")
      .merge(labs_ord,   on="hadm_id",  how="left")
      .merge(vitals_ord, on="hadm_id",  how="left"))

meta_cols = ["hadm_id", "subject_id"] + TASKS
feat_cols = [c for c in df.columns if c not in meta_cols]

# Classify each feature into a group
static_feats   = [c for c in feat_cols if c in static.columns]
ordering_feats = [c for c in feat_cols if "__lab" in c and (
                  "_obs_" in c or "ordering_" in c)
                  or "__vit" in c and ("_obs_" in c or "ordering_" in c)]
value_feats    = [c for c in feat_cols
                  if c not in static_feats and c not in ordering_feats]

print(f"\nFeature groups:")
print(f"  Static   : {len(static_feats)}")
print(f"  Values   : {len(value_feats)}")
print(f"  Ordering : {len(ordering_feats)}")


# ── 2. Patient split (same seed) ─────────────────────────────────────────────
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


# ── 3. Phenotype labels (deceptively normal, concordant, etc.) ───────────────
val_mat = df[value_feats].astype("float32")
means, stds = val_mat.mean(), val_mat.std().replace(0, 1)
df["value_abnormality"]      = ((val_mat - means) / stds).abs().mean(axis=1)
oi_cols = [c for c in df.columns if "ordering_intensity" in c]
df["ordering_intensity_all"] = df[oi_cols].mean(axis=1)

df["val_q"] = pd.qcut(df["value_abnormality"],      4, labels=[0,1,2,3])
df["ord_q"] = pd.qcut(df["ordering_intensity_all"], 4, labels=[0,1,2,3])

phenotype_map = {
    (0, 3): "deceptively_normal",
    (0, 0): "concordant_normal",
    (3, 3): "concordant_abnormal",
    (3, 0): "contradictory",       # high values, low ordering
}
df["phenotype"] = df.apply(
    lambda r: phenotype_map.get((r["val_q"], r["ord_q"]), "other"), axis=1)
print(f"\nPhenotypes:\n{df['phenotype'].value_counts()}")


# ── 4. SHAP analysis per task ─────────────────────────────────────────────────
all_shap_results = {}

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
    prev = y_tr[ok_tr].mean()
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        scale_pos_weight=(1-prev)/prev,
        n_jobs=-1, random_state=SEED, verbose=-1, metric="auc",
    )
    model.fit(X_tr[ok_tr], y_tr[ok_tr],
              eval_set=[(X_va[ok_va], y_va[ok_va])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    use_booster = False

    # SHAP on test set (sample 5000 for speed if large)
    X_shap = X_te[ok_te].reset_index(drop=True)
    meta_shap = df.loc[te_m][ok_te][["hadm_id", "phenotype"]].reset_index(drop=True)

    if len(X_shap) > 5000:
        idx = np.random.RandomState(SEED).choice(len(X_shap), 5000, replace=False)
        X_shap    = X_shap.iloc[idx]
        meta_shap = meta_shap.iloc[idx]

    print(f"  Computing SHAP values for {len(X_shap):,} test patients...")
    if use_booster:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_shap)          # shape (N, F)
    else:
        explainer = shap.TreeExplainer(model.booster_)
        sv = explainer.shap_values(X_shap)

    # Absolute SHAP per feature group per patient
    feat_arr = np.array(feat_cols)
    static_idx   = [i for i, c in enumerate(feat_cols) if c in set(static_feats)]
    value_idx    = [i for i, c in enumerate(feat_cols) if c in set(value_feats)]
    ordering_idx = [i for i, c in enumerate(feat_cols) if c in set(ordering_feats)]

    shap_abs = np.abs(sv)
    attr = pd.DataFrame({
        "hadm_id"         : meta_shap["hadm_id"].values,
        "phenotype"       : meta_shap["phenotype"].values,
        "shap_static"     : shap_abs[:, static_idx].sum(axis=1),
        "shap_values"     : shap_abs[:, value_idx].sum(axis=1),
        "shap_ordering"   : shap_abs[:, ordering_idx].sum(axis=1),
    })
    attr["total_shap"] = attr[["shap_static","shap_values","shap_ordering"]].sum(axis=1)
    attr["frac_ordering"] = attr["shap_ordering"] / attr["total_shap"].replace(0, np.nan)
    attr["frac_values"]   = attr["shap_values"]   / attr["total_shap"].replace(0, np.nan)
    attr["frac_static"]   = attr["shap_static"]   / attr["total_shap"].replace(0, np.nan)
    attr.to_csv(OUT / f"shap_group_attributions_{task}.csv", index=False)

    # Mean attribution by phenotype
    pheno_summary = attr.groupby("phenotype")[
        ["frac_static","frac_values","frac_ordering"]].mean().round(4)
    pheno_summary.to_csv(OUT / f"phenotype_attribution_{task}.csv")

    print(f"\n  Mean SHAP fraction by phenotype (task={task}):")
    print(f"  {'Phenotype':<25} {'Static':>8} {'Values':>8} {'Ordering':>10}")
    print(f"  {'-'*55}")
    for ph, row in pheno_summary.iterrows():
        print(f"  {ph:<25} {row['frac_static']:>8.3f} "
              f"{row['frac_values']:>8.3f} {row['frac_ordering']:>10.3f}")

    # Top features overall
    mean_abs_shap = pd.Series(shap_abs.mean(axis=0), index=feat_cols)
    top = mean_abs_shap.sort_values(ascending=False).head(30)
    top.to_csv(OUT / f"shap_top_features_{task}.csv", header=["mean_abs_shap"])

    all_shap_results[task] = {
        "phenotype_attribution": pheno_summary.to_dict(),
        "top_feature": top.index[0],
        "ordering_mean_frac": float(attr["frac_ordering"].mean()),
        "ordering_frac_deceptively_normal": float(
            attr.loc[attr["phenotype"]=="deceptively_normal", "frac_ordering"].mean())
            if "deceptively_normal" in attr["phenotype"].values else None,
    }

with open(OUT / "shap_results.json", "w") as f:
    json.dump(all_shap_results, f, indent=2, default=str)

print(f"\nAll results saved to {OUT}/")
