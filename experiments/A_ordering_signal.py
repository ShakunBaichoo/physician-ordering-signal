#!/usr/bin/env python
"""
Experiment A — Test-Ordering Patterns as an Independent Predictive Signal
=========================================================================
Tests the hypothesis that lab/vital ordering frequencies (how many times each
test was measured per 4h bin) encode physician acuity assessment independently
of the test values themselves.

Three LightGBM models compared:
  1. values_only   — standard lab/vital values
  2. ordering_only — ordering frequencies (_obs columns) only
  3. combined      — values + ordering features

Key sub-analysis: "Deceptively Normal" patients
  Patients with normal lab values but HIGH ordering intensity → do they have
  worse outcomes? If yes, physicians are detecting something values alone miss.

Outputs (results/novel/A_ordering_signal/):
  - metrics.json          AUROC/AUPRC for all 3 models × 4 tasks
  - ordering_features.csv derived ordering metrics per admission
  - deceptively_normal/   analysis of value-ordering discordant patients
  - lgbm_<model>_<task>   saved models
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "1_ordering_paper" / "results" / "A_ordering_signal"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "deceptively_normal").mkdir(exist_ok=True)

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED  = 42


# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
cohort = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
labels = pd.read_parquet(DATA / "labels.parquet")
static = pd.read_parquet(DATA / "static.parquet")

print("  Loading labs time-series...")
labs_long = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
print("  Loading vitals time-series...")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")


# ── 2. Build feature sets ─────────────────────────────────────────────────────

def extract_value_features(df_long: pd.DataFrame, name: str) -> pd.DataFrame:
    """Aggregate value columns (mean/max/min/last) across 12 time bins."""
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id", "time_bin")
                and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()


def extract_ordering_features(df_long: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Build rich ordering features from _obs columns:
      - total_obs_{feat}     : total measurements across 48h
      - intensity_{feat}     : fraction of 12 bins where feat was measured
      - ordering_slope_{feat}: linear trend in obs over time (increasing = escalating concern)
    Admission-level summaries:
      - ordering_intensity   : mean fraction of labs ordered per bin
      - ordering_diversity   : n distinct labs ordered at least once
      - ordering_escalation  : slope of total-labs-ordered over time bins
    """
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    if not obs_cols:
        return pd.DataFrame({"hadm_id": df_long["hadm_id"].unique()})

    print(f"  Computing ordering features for {name} ({len(obs_cols)} tests)...")

    rows = []
    for hadm_id, grp in df_long.groupby("hadm_id", sort=False):
        grp = grp.sort_values("time_bin")
        obs = grp[obs_cols].values          # (n_bins, n_feats)
        n_bins = obs.shape[0]
        t = np.arange(n_bins)

        feat = {"hadm_id": hadm_id}

        # Per-test features
        for i, col in enumerate(obs_cols):
            base = col[:-4]  # strip _obs
            vals = obs[:, i].astype(float)
            feat[f"total_obs_{base}__{name}"] = vals.sum()
            feat[f"intensity_{base}__{name}"]  = (vals > 0).mean()
            # linear slope of ordering over time
            if n_bins > 1 and vals.sum() > 0:
                slope = np.polyfit(t, vals, 1)[0]
            else:
                slope = 0.0
            feat[f"slope_{base}__{name}"] = slope

        # Admission-level ordering summaries
        binary = (obs > 0).astype(float)   # was each test ordered in each bin?
        tests_per_bin = binary.sum(axis=1)  # total distinct tests per bin
        feat[f"ordering_intensity__{name}"]  = binary.mean()          # mean frac ordered
        feat[f"ordering_diversity__{name}"]  = (obs.sum(axis=0) > 0).sum()  # n tests used at all
        feat[f"ordering_breadth__{name}"]    = tests_per_bin.mean()   # mean tests/bin
        if n_bins > 1:
            feat[f"ordering_escalation__{name}"] = np.polyfit(t, tests_per_bin, 1)[0]
        else:
            feat[f"ordering_escalation__{name}"] = 0.0

        rows.append(feat)

    return pd.DataFrame(rows)


print("\nExtracting value features...")
labs_val   = extract_value_features(labs_long,   "lab")
vitals_val = extract_value_features(vitals_long, "vit")

print("\nExtracting ordering features...")
labs_ord   = extract_ordering_features(labs_long,   "lab")
vitals_ord = extract_ordering_features(vitals_long, "vit")


# ── 3. Merge ──────────────────────────────────────────────────────────────────
print("\nMerging...")
base = cohort.merge(labels, on=["hadm_id", "subject_id"])

def build_feature_df(include_values=True, include_ordering=True) -> pd.DataFrame:
    df = base.copy()
    df = df.merge(static, on="hadm_id", how="left")
    if include_values:
        df = df.merge(labs_val,   on="hadm_id", how="left")
        df = df.merge(vitals_val, on="hadm_id", how="left")
    if include_ordering:
        df = df.merge(labs_ord,   on="hadm_id", how="left")
        df = df.merge(vitals_ord, on="hadm_id", how="left")
    return df

meta_cols = ["hadm_id", "subject_id"] + TASKS
df_val  = build_feature_df(include_values=True,  include_ordering=False)
df_ord  = build_feature_df(include_values=False, include_ordering=True)
df_both = build_feature_df(include_values=True,  include_ordering=True)

print(f"  Values-only  : {len([c for c in df_val.columns  if c not in meta_cols])} features")
print(f"  Ordering-only: {len([c for c in df_ord.columns  if c not in meta_cols])} features")
print(f"  Combined     : {len([c for c in df_both.columns if c not in meta_cols])} features")

# Save ordering features for downstream analysis
ord_feat_cols = [c for c in df_both.columns if "ordering_intensity" in c
                 or "ordering_diversity" in c or "ordering_escalation" in c
                 or "ordering_breadth" in c]
df_both[["hadm_id"] + ord_feat_cols].to_csv(OUT / "ordering_features.csv", index=False)


# ── 4. Patient-level split ────────────────────────────────────────────────────
def patient_split(df):
    pat = df.groupby("subject_id")["mortality"].max().reset_index()
    pat = pat.sample(frac=1, random_state=SEED)
    n = len(pat)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    train_s = set(pat.iloc[:n_train]["subject_id"])
    val_s   = set(pat.iloc[n_train:n_train + n_val]["subject_id"])
    test_s  = set(pat.iloc[n_train + n_val:]["subject_id"])
    return (df["subject_id"].isin(train_s),
            df["subject_id"].isin(val_s),
            df["subject_id"].isin(test_s))

tr_m, va_m, te_m = patient_split(df_both)  # same split across all three datasets
print(f"\nSplit — Train {tr_m.sum():,} | Val {va_m.sum():,} | Test {te_m.sum():,}")


# ── 5. Train & evaluate ───────────────────────────────────────────────────────
def train_eval(df, model_name, task, tr, va, te):
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X_tr = df.loc[tr, feat_cols].astype("float32")
    X_va = df.loc[va, feat_cols].astype("float32")
    X_te = df.loc[te, feat_cols].astype("float32")
    y_tr, y_va, y_te = df.loc[tr, task], df.loc[va, task], df.loc[te, task]

    ok_tr, ok_va, ok_te = y_tr.notna(), y_va.notna(), y_te.notna()
    prev = y_tr[ok_tr].mean()
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

    def score(X, y, split):
        prob  = model.predict_proba(X)[:, 1]
        return {"auroc": round(roc_auc_score(y, prob), 4),
                "auprc": round(average_precision_score(y, prob), 4),
                "probs": prob.tolist() if split == "test" else None}

    val_r  = score(X_va[ok_va], y_va[ok_va], "val")
    test_r = score(X_te[ok_te], y_te[ok_te], "test")
    model.booster_.save_model(str(OUT / f"lgbm_{model_name}_{task}.txt"))

    return val_r, test_r, model, feat_cols, X_te[ok_te], y_te[ok_te]


all_results = {m: {} for m in ["values_only", "ordering_only", "combined"]}

datasets = {
    "values_only" : df_val,
    "ordering_only": df_ord,
    "combined"    : df_both,
}

# Store test probs for combined model (used in conformal analysis)
combined_test_probs = {}
combined_test_labels = {}

for model_name, df in datasets.items():
    # align masks to this df's index
    tr = df["subject_id"].isin(
        df_both.loc[tr_m, "subject_id"])
    va = df["subject_id"].isin(
        df_both.loc[va_m, "subject_id"])
    te = df["subject_id"].isin(
        df_both.loc[te_m, "subject_id"])

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    for task in TASKS:
        print(f"\n  Task: {task}")
        val_r, test_r, model, feat_cols, X_te, y_te = train_eval(
            df, model_name, task, tr, va, te)
        print(f"    Val  — AUROC {val_r['auroc']:.4f} | AUPRC {val_r['auprc']:.4f}")
        print(f"    Test — AUROC {test_r['auroc']:.4f} | AUPRC {test_r['auprc']:.4f}")
        all_results[model_name][task] = {"val": val_r, "test": test_r}

        if model_name == "combined":
            prob = model.predict_proba(X_te)[:, 1]
            combined_test_probs[task]  = prob
            combined_test_labels[task] = y_te.values


# ── 6. Summary table ──────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("  AUROC COMPARISON (Test Set)")
print(f"{'='*70}")
print(f"  {'Task':<22} {'Values':>8} {'Ordering':>10} {'Combined':>10} {'Delta':>8}")
print(f"  {'-'*62}")
for task in TASKS:
    v = all_results["values_only"][task]["test"]["auroc"]
    o = all_results["ordering_only"][task]["test"]["auroc"]
    c = all_results["combined"][task]["test"]["auroc"]
    delta = c - v
    print(f"  {task:<22} {v:>8.4f} {o:>10.4f} {c:>10.4f} {delta:>+8.4f}")


# ── 7. Deceptively Normal Analysis ───────────────────────────────────────────
print("\n\nDeceptively Normal Patient Analysis...")

# Compute a composite "value normality score" from the labs (z-score from population)
val_feat_cols = [c for c in df_val.columns if c not in meta_cols]
X_all = df_both[val_feat_cols].astype("float32")
means = X_all.mean()
stds  = X_all.std().replace(0, 1)
z_scores = ((X_all - means) / stds).abs()
df_both["value_abnormality"] = z_scores.mean(axis=1)  # higher = more abnormal

# Compute overall ordering intensity (mean of lab + vital ordering intensity cols)
oi_cols = [c for c in df_both.columns if "ordering_intensity" in c]
df_both["ordering_intensity_overall"] = df_both[oi_cols].mean(axis=1)

# Quartile-based stratification
df_both["val_q"]  = pd.qcut(df_both["value_abnormality"],     4, labels=[0,1,2,3])
df_both["ord_q"]  = pd.qcut(df_both["ordering_intensity_overall"], 4, labels=[0,1,2,3])

# "Deceptively Normal" = low value abnormality (Q1) + high ordering (Q4)
dn_mask = (df_both["val_q"] == 0) & (df_both["ord_q"] == 3)
# "Concordant Normal" = low value abnormality + low ordering
cn_mask = (df_both["val_q"] == 0) & (df_both["ord_q"] == 0)
# "Concordant Abnormal" = high value abnormality + high ordering
ca_mask = (df_both["val_q"] == 3) & (df_both["ord_q"] == 3)

groups = {
    "deceptively_normal (low_values+high_ordering)": dn_mask,
    "concordant_normal  (low_values+low_ordering) ": cn_mask,
    "concordant_abnormal(high_values+high_ordering)": ca_mask,
}

print(f"\n  {'Group':<48} {'N':>7}  ", end="")
for t in TASKS:
    print(f"  {t[:8]:>8}", end="")
print()
print(f"  {'-'*90}")

dn_results = {}
for gname, mask in groups.items():
    sub = df_both[mask]
    n = len(sub)
    rates = {t: sub[t].mean() for t in TASKS}
    print(f"  {gname:<48} {n:>7}", end="")
    for t in TASKS:
        print(f"  {rates[t]:>8.3f}", end="")
    print()
    dn_results[gname.split("(")[0].strip()] = {"n": n, "rates": {t: round(rates[t], 4) for t in TASKS}}

df_both[["hadm_id", "value_abnormality", "ordering_intensity_overall",
         "val_q", "ord_q"] + TASKS].to_csv(
    OUT / "deceptively_normal" / "patient_stratification.csv", index=False)

# Save everything
with open(OUT / "metrics.json", "w") as f:
    json.dump({"model_comparison": all_results,
               "deceptively_normal": dn_results}, f, indent=2, default=str)

print(f"\nAll results saved to {OUT}/")
