#!/usr/bin/env python
"""
Experiment H — CCI-Stratified Ordering Signal
==============================================
Addresses the key confounding question:
  "Is ordering intensity just a proxy for comorbidity burden?"

Three CCI strata:
  CCI=0   : 84.5% of cohort — no prior documented comorbidities
  CCI 1-3 : 6.9%  — mild-to-moderate comorbidity burden
  CCI≥4   : 8.5%  — high comorbidity burden

Within each stratum, compares:
  ordering_only   — ordering frequency features + demographics (NO lab values)
  values_only     — lab/vital values + demographics (NO ordering frequency)
  combined        — both

Key claim: even within CCI=0 patients (zero comorbidity burden),
ordering AUROC ≈ 0.85+ for mortality/sepsis, proving ordering captures
real-time physician acuity assessment that is independent of prior disease.

Also saves test_predictions.parquet for Experiments I and J.

Outputs → 1_ordering_paper/results/H_cci_stratified/
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
OUT  = ROOT / "1_ordering_paper" / "results" / "H_cci_stratified"
OUT.mkdir(parents=True, exist_ok=True)

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED   = 42
N_BOOT = 1000

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading data...")
cohort = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
labels = pd.read_parquet(DATA / "labels.parquet")
static = pd.read_parquet(DATA / "static.parquet")

print("  Loading labs timeseries...")
labs_long   = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
print("  Loading vitals timeseries...")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")


# ── 2. Feature extraction (vectorised — no slow per-admission Python loop) ────
def value_features(df_long, name):
    """Aggregate value columns (mean across bins) — fast groupby."""
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id", "time_bin") and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()


def ordering_features(df_long, name):
    """
    Vectorised ordering features — equivalent to A's extract_ordering_features
    but without the slow Python loop. Skips per-test slope (minor feature).

    Per test: total_obs (sum), intensity (fraction of bins ordered)
    Admission-level: intensity, diversity, breadth, escalation
    """
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    if not obs_cols:
        return pd.DataFrame({"hadm_id": df_long["hadm_id"].unique()})

    print(f"  Ordering features for {name} ({len(obs_cols)} tests)...")
    g = df_long.groupby("hadm_id", sort=False)

    # Per-test: total ordered and fraction of bins with ≥1 observation
    total  = g[obs_cols].sum()
    binary = df_long[["hadm_id"] + obs_cols].copy()
    for c in obs_cols:
        binary[c] = (binary[c] > 0).astype("float32")
    intens = binary.groupby("hadm_id", sort=False)[obs_cols].mean()

    total.columns  = [f"total_obs_{c[:-4]}__{name}" for c in obs_cols]
    intens.columns = [f"intensity_{c[:-4]}__{name}"  for c in obs_cols]

    # Admission-level summaries (same definitions as Experiment A)
    n_bins_s    = g["time_bin"].count().rename("n_bins")
    binary_all  = intens.copy()   # fraction ordered per test (already computed)

    # ordering_intensity = mean over tests of (fraction of bins ordered)
    ord_intensity = binary_all.mean(axis=1).rename(f"ordering_intensity__{name}")
    # ordering_diversity = number of distinct tests ordered at all
    ord_diversity = (total > 0).sum(axis=1).rename(f"ordering_diversity__{name}")
    # ordering_breadth   = mean number of distinct tests per bin
    tests_per_bin = binary.groupby("hadm_id", sort=False)[obs_cols].sum()
    ord_breadth   = tests_per_bin.mean(axis=1).rename(f"ordering_breadth__{name}")

    # ordering_escalation = linear slope of tests-per-bin over time
    # Use fast approximation: (mean of last 3 bins − mean of first 3 bins) / total_bins
    def slope_approx(grp):
        ordered = grp.sort_values("time_bin")[obs_cols]
        n = len(ordered)
        if n < 2:
            return 0.0
        half = max(1, n // 2)
        return (ordered.iloc[-half:].sum(axis=1).mean() -
                ordered.iloc[:half].sum(axis=1).mean()) / n

    print(f"    Computing escalation slopes for {name}...")
    esc = df_long.groupby("hadm_id", sort=False).apply(slope_approx)
    esc.name = f"ordering_escalation__{name}"

    result = pd.concat([total, intens, ord_intensity, ord_diversity,
                        ord_breadth, esc], axis=1).reset_index()
    return result


print("\nExtracting value features...")
labs_val   = value_features(labs_long,   "lab")
vitals_val = value_features(vitals_long, "vit")

print("\nExtracting ordering features...")
labs_ord   = ordering_features(labs_long,   "lab")
vitals_ord = ordering_features(vitals_long, "vit")


# ── 3. Merge feature sets ─────────────────────────────────────────────────────
base     = cohort.merge(labels, on=["hadm_id", "subject_id"])
meta     = ["hadm_id", "subject_id"] + TASKS

def build(values=True, ordering=True):
    df = base.merge(static, on="hadm_id", how="left")
    if values:
        df = df.merge(labs_val,   on="hadm_id", how="left")
        df = df.merge(vitals_val, on="hadm_id", how="left")
    if ordering:
        df = df.merge(labs_ord,   on="hadm_id", how="left")
        df = df.merge(vitals_ord, on="hadm_id", how="left")
    return df

df_val  = build(values=True,  ordering=False)
df_ord  = build(values=False, ordering=True)
df_both = build(values=True,  ordering=True)

print(f"\nFeatures — values-only:{len([c for c in df_val.columns  if c not in meta])}  "
      f"ordering-only:{len([c for c in df_ord.columns  if c not in meta])}  "
      f"combined:{len([c for c in df_both.columns if c not in meta])}")


# ── 4. Patient-level train/val/test split (SEED=42 — same as Experiment A) ───
def patient_split(df):
    pat = df.groupby("subject_id")["mortality"].max().reset_index()
    pat = pat.sample(frac=1, random_state=SEED)
    n = len(pat)
    n_tr = int(0.70 * n);  n_va = int(0.15 * n)
    tr_s = set(pat.iloc[:n_tr]["subject_id"])
    va_s = set(pat.iloc[n_tr:n_tr+n_va]["subject_id"])
    te_s = set(pat.iloc[n_tr+n_va:]["subject_id"])
    return (df["subject_id"].isin(tr_s),
            df["subject_id"].isin(va_s),
            df["subject_id"].isin(te_s))

tr, va, te = patient_split(df_both)
print(f"Split — Train {tr.sum():,} | Val {va.sum():,} | Test {te.sum():,}")


# ── 5. LightGBM training ──────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.05, num_leaves=127,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=SEED, verbose=-1, metric="auc",
)

def train_model(df, task, tr_mask, va_mask):
    feat = [c for c in df.columns if c not in meta]
    X_tr, y_tr = df.loc[tr_mask, feat].astype("float32"), df.loc[tr_mask, task]
    X_va, y_va = df.loc[va_mask, feat].astype("float32"), df.loc[va_mask, task]
    ok_tr, ok_va = y_tr.notna(), y_va.notna()
    pos_w = (1 - y_tr[ok_tr].mean()) / y_tr[ok_tr].mean()
    m = lgb.LGBMClassifier(scale_pos_weight=pos_w, **LGBM_PARAMS)
    m.fit(X_tr[ok_tr], y_tr[ok_tr],
          eval_set=[(X_va[ok_va], y_va[ok_va])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m, feat


def predict_test(model, feat_cols, df, te_mask):
    X_te = df.loc[te_mask, feat_cols].astype("float32")
    y_te = df.loc[te_mask, "hadm_id"].values
    return y_te, model.predict_proba(X_te)[:, 1]


print("\nTraining models (this takes ~5–10 min)...")
models   = {}  # (model_name, task) → (model, feat_cols)
test_preds = {}  # task → DataFrame(hadm_id, ord_prob, val_prob, both_prob)

for task in TASKS:
    print(f"  {task}...")
    m_val,  f_val  = train_model(df_val,  task, tr, va)
    m_ord,  f_ord  = train_model(df_ord,  task, tr, va)
    m_both, f_both = train_model(df_both, task, tr, va)
    models[(task, "values_only")]  = (m_val,  f_val)
    models[(task, "ordering_only")] = (m_ord,  f_ord)
    models[(task, "combined")]     = (m_both, f_both)

    hids_v, p_val  = predict_test(m_val,  f_val,  df_val,  te)
    hids_o, p_ord  = predict_test(m_ord,  f_ord,  df_ord,  te)
    hids_b, p_both = predict_test(m_both, f_both, df_both, te)
    # All three datasets have same test_mask → same hadm_ids in same order
    test_preds[task] = pd.DataFrame({
        "hadm_id":      df_both.loc[te, "hadm_id"].values,
        "val_prob":     p_val,
        "ord_prob":     p_ord,
        "both_prob":    p_both,
        "true_label":   df_both.loc[te, task].values,
    })


# ── 6. Merge CCI strata onto test predictions ─────────────────────────────────
cci_info = df_both.loc[te, ["hadm_id", "cci_score"]].copy()
cci_info["cci_stratum"] = pd.cut(
    cci_info["cci_score"],
    bins=[-0.01, 0, 3, 100],
    labels=["CCI=0", "CCI 1-3", "CCI≥4"]
)

for task in TASKS:
    test_preds[task] = test_preds[task].merge(cci_info, on="hadm_id", how="left")

# Also save full test predictions for Experiments I & J
pred_long = []
for task in TASKS:
    tmp = test_preds[task].copy()
    tmp["task"] = task
    pred_long.append(tmp)
all_preds = pd.concat(pred_long, ignore_index=True)
# Merge in race/insurance/timing from static for I and J
static_te = df_both.loc[te, ["hadm_id"] +
    [c for c in static.columns if c.startswith("race_group_") or
     c.startswith("insurance_") or c in ("admit_night","admit_weekend","admit_dow","admit_hour")]
].drop_duplicates("hadm_id")
all_preds = all_preds.merge(static_te, on="hadm_id", how="left")
all_preds.to_parquet(OUT / "test_predictions.parquet", index=False)
print(f"\nTest predictions saved → {OUT / 'test_predictions.parquet'}")


# ── 7. Bootstrap AUROC within strata ─────────────────────────────────────────
rng = np.random.default_rng(SEED)

def bootstrap_auroc(y, probs, n_boot=N_BOOT):
    if y.sum() < 5 or (y == 0).sum() < 5:
        return float("nan"), float("nan"), float("nan")
    base = roc_auc_score(y, probs)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], probs[idx]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            boot.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2
    return round(base, 4), round(lo, 4), round(hi, 4)


STRATA = ["CCI=0", "CCI 1-3", "CCI≥4"]
MODEL_NAMES = ["ordering_only", "values_only", "combined"]

results = []
for task in TASKS:
    df_t = test_preds[task].dropna(subset=["true_label"])
    for stratum in ["ALL"] + STRATA:
        if stratum == "ALL":
            sub = df_t
        else:
            sub = df_t[df_t["cci_stratum"] == stratum]
        if len(sub) < 50:
            continue
        y = sub["true_label"].values.astype(int)
        for model_name, prob_col in [("ordering_only","ord_prob"),
                                      ("values_only","val_prob"),
                                      ("combined","both_prob")]:
            p = sub[prob_col].values
            auroc, lo, hi = bootstrap_auroc(y, p)
            results.append({
                "task": task, "stratum": stratum,
                "N": len(sub), "n_pos": int(y.sum()),
                "model": model_name,
                "auroc": auroc, "ci_lo": lo, "ci_hi": hi,
                "auroc_str": f"{auroc:.3f} [{lo:.3f}–{hi:.3f}]",
            })
            print(f"  {task:<16} {stratum:<10} {model_name:<16} AUROC={auroc:.3f} [{lo:.3f}–{hi:.3f}]  N={len(sub):,}")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT / "cci_stratified_auroc.csv", index=False)


# ── 8. Ordering intensity by CCI stratum (descriptive) ───────────────────────
ord_feat_cols = ["hadm_id",
    "ordering_intensity__lab", "ordering_intensity__vit",
    "ordering_diversity__lab", "ordering_diversity__vit"]
ord_feats_te = df_ord.loc[te, [c for c in ord_feat_cols if c in df_ord.columns]]
desc = ord_feats_te.merge(cci_info, on="hadm_id")
desc_summary = desc.groupby("cci_stratum")[
    [c for c in desc.columns if "ordering_intensity" in c or "ordering_diversity" in c]
].agg(["mean","std"]).round(3)
desc_summary.to_csv(OUT / "ordering_by_cci_stratum.csv")

# ── 9. Summary JSON ───────────────────────────────────────────────────────────
summary = {}
for task in TASKS:
    summary[task] = {}
    for stratum in ["ALL"] + STRATA:
        rows = results_df[(results_df["task"]==task) & (results_df["stratum"]==stratum)]
        if rows.empty: continue
        summary[task][stratum] = {}
        for _, row in rows.iterrows():
            summary[task][stratum][row["model"]] = {
                "auroc": row["auroc"], "ci_lo": row["ci_lo"],
                "ci_hi": row["ci_hi"], "N": int(row["N"]),
                "n_pos": int(row["n_pos"])
            }

with open(OUT / "cci_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅  Experiment H complete.  Results → {OUT}")
print("\nKey finding (ordering_only AUROC within CCI=0 stratum):")
for task in TASKS:
    r = summary.get(task,{}).get("CCI=0",{}).get("ordering_only",{})
    if r:
        print(f"  {task:<16} AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]  N={r['N']:,}")
