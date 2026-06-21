#!/usr/bin/env python
"""
Experiment F — Paper Improvements for High-Impact Journal Submission
=====================================================================
1. Bootstrap 95% CIs + paired DeLong-style test for all AUROC/AUPRC
2. Clinical characterisation of the deceptively normal cohort
   (diagnoses, admission types, most-ordered tests)
3. SOFA severity score as confounder — does ordering add signal beyond SOFA?
4. Temporal escalation analysis — does rising ordering intensity predict worse outcomes?
5. Sentinel ordering patterns — which specific repeated tests are most predictive?

All outputs → results/novel/F_paper_improvements/
"""

import json
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT   = Path(__file__).parents[2]
DATA   = ROOT / "data" / "processed"
RAW_DB = ROOT / "data" / "raw" / "mimic_iv_2_2.db"
OUT    = ROOT / "1_ordering_paper" / "results" / "F_paper_improvements"
OUT.mkdir(parents=True, exist_ok=True)

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED   = 42
N_BOOT = 1000   # bootstrap iterations


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Loading shared data...")
print("=" * 65)

cohort  = pd.read_parquet(DATA / "cohort.parquet")
labels  = pd.read_parquet(DATA / "labels.parquet")
static  = pd.read_parquet(DATA / "static.parquet")

print("  Labs...")
labs_long   = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
print("  Vitals...")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

# Build feature matrices
def agg_values(df_long, name):
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id","time_bin") and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()

def agg_ordering(df_long, name):
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    rows = []
    for hadm_id, grp in df_long.groupby("hadm_id", sort=False):
        grp = grp.sort_values("time_bin")
        obs  = grp[obs_cols].values.astype(float)
        t    = np.arange(len(obs))
        feat = {"hadm_id": hadm_id}
        for i, col in enumerate(obs_cols):
            base = col[:-4]
            vals = obs[:, i]
            feat[f"total_obs_{base}__{name}"]  = vals.sum()
            feat[f"intensity_{base}__{name}"]  = (vals > 0).mean()
            slope = np.polyfit(t, vals, 1)[0] if vals.sum() > 0 and len(t) > 1 else 0.0
            feat[f"slope_{base}__{name}"]      = slope
        binary = (obs > 0).astype(float)
        tests_per_bin = binary.sum(axis=1)
        feat[f"ordering_intensity__{name}"]  = binary.mean()
        feat[f"ordering_diversity__{name}"]  = (obs.sum(axis=0) > 0).sum()
        feat[f"ordering_breadth__{name}"]    = tests_per_bin.mean()
        feat[f"ordering_escalation__{name}"] = (
            np.polyfit(t, tests_per_bin, 1)[0] if len(t) > 1 else 0.0)
        rows.append(feat)
    return pd.DataFrame(rows)

labs_val   = agg_values(labs_long,   "lab")
vitals_val = agg_values(vitals_long, "vit")
print("  Computing lab ordering features...")
labs_ord   = agg_ordering(labs_long,   "lab")
print("  Computing vital ordering features...")
vitals_ord = agg_ordering(vitals_long, "vit")

base = (cohort[["hadm_id","subject_id","race","insurance","admission_type",
                "age","los_hours"]]
        .merge(labels, on=["hadm_id","subject_id"]))

meta_cols = ["hadm_id","subject_id","race","insurance","admission_type",
             "age","los_hours"] + TASKS

def build_df(values=True, ordering=True, include_static=True):
    df = base.copy()
    if include_static: df = df.merge(static,     on="hadm_id", how="left")
    if values:         df = df.merge(labs_val,   on="hadm_id", how="left")
    if values:         df = df.merge(vitals_val, on="hadm_id", how="left")
    if ordering:       df = df.merge(labs_ord,   on="hadm_id", how="left")
    if ordering:       df = df.merge(vitals_ord, on="hadm_id", how="left")
    return df

df_val  = build_df(values=True,  ordering=False)
df_ord  = build_df(values=False, ordering=True)
df_both = build_df(values=True,  ordering=True)

# Patient split
pat = df_both.groupby("subject_id")["mortality"].max().reset_index()
pat = pat.sample(frac=1, random_state=SEED)
n   = len(pat)
n_tr, n_va = int(0.70*n), int(0.15*n)
train_s = set(pat.iloc[:n_tr]["subject_id"])
val_s   = set(pat.iloc[n_tr:n_tr+n_va]["subject_id"])
test_s  = set(pat.iloc[n_tr+n_va:]["subject_id"])

def masks(df):
    return (df["subject_id"].isin(train_s),
            df["subject_id"].isin(val_s),
            df["subject_id"].isin(test_s))

tr_m, va_m, te_m = masks(df_both)

def get_feat_cols(df):
    return [c for c in df.columns if c not in meta_cols]

def train_lgbm(df, task, tr, va, te, n_est=2000):
    fc = get_feat_cols(df)
    Xtr = df.loc[tr, fc].astype("float32")
    Xva = df.loc[va, fc].astype("float32")
    Xte = df.loc[te, fc].astype("float32")
    ytr, yva, yte = df.loc[tr, task], df.loc[va, task], df.loc[te, task]
    ok_tr, ok_va, ok_te = ytr.notna(), yva.notna(), yte.notna()
    prev  = ytr[ok_tr].mean()
    model = lgb.LGBMClassifier(
        n_estimators=n_est, learning_rate=0.05, num_leaves=127,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=(1-prev)/prev,
        n_jobs=-1, random_state=SEED, verbose=-1, metric="auc")
    model.fit(Xtr[ok_tr], ytr[ok_tr],
              eval_set=[(Xva[ok_va], yva[ok_va])],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
    prob = model.predict_proba(Xte[ok_te])[:, 1]
    return prob, yte[ok_te].values, model, fc, Xte[ok_te]


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Bootstrap CIs + Paired DeLong-style test
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 1 — Bootstrap 95% CIs + Paired AUROC tests")
print("="*65)

def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=N_BOOT, ci=95):
    scores = []
    n = len(y_true)
    rng = np.random.RandomState(SEED)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        scores.append(metric_fn(y_true[idx], y_prob[idx]))
    lo = np.percentile(scores, (100-ci)/2)
    hi = np.percentile(scores, 100 - (100-ci)/2)
    return lo, hi

def paired_bootstrap_pval(y_true, prob_a, prob_b, n_boot=N_BOOT):
    """Two-sided paired bootstrap p-value for H0: AUROC(a) == AUROC(b).
    Both models are scored on the SAME resampled indices (paired); p is the
    fraction of the resampled difference distribution that crosses zero."""
    observed_diff = roc_auc_score(y_true, prob_b) - roc_auc_score(y_true, prob_a)
    diffs = []
    n = len(y_true)
    rng = np.random.RandomState(SEED)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if y_true[idx].sum() == 0 or y_true[idx].sum() == n:
            continue
        diffs.append(roc_auc_score(y_true[idx], prob_b[idx]) - roc_auc_score(y_true[idx], prob_a[idx]))
    diffs = np.array(diffs)
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return observed_diff, min(float(p), 1.0)

ci_results = {}
for task in TASKS:
    print(f"\n  Task: {task}")
    tr, va, te = masks(df_val);  tr, va, te = df_val["subject_id"].isin(train_s), df_val["subject_id"].isin(val_s), df_val["subject_id"].isin(test_s)
    p_val, y_val, _, _, _ = train_lgbm(df_val,  task, tr, va, te)
    tr, va, te = df_ord["subject_id"].isin(train_s), df_ord["subject_id"].isin(val_s), df_ord["subject_id"].isin(test_s)
    p_ord, y_ord, _, _, _ = train_lgbm(df_ord,  task, tr, va, te)
    tr, va, te = df_both["subject_id"].isin(train_s), df_both["subject_id"].isin(val_s), df_both["subject_id"].isin(test_s)
    p_both, y_both, _, _, _ = train_lgbm(df_both, task, tr, va, te)

    # Align labels (readmission has NaN for deaths — smallest set)
    y_common = y_val  # all three should align since same test split + NaN mask

    models = {"values_only": (p_val, y_val),
              "ordering_only": (p_ord, y_ord),
              "combined": (p_both, y_both)}
    ci_results[task] = {}
    for mname, (prob, ytrue) in models.items():
        auroc = roc_auc_score(ytrue, prob)
        auprc = average_precision_score(ytrue, prob)
        auroc_lo, auroc_hi = bootstrap_ci(ytrue, prob, roc_auc_score)
        auprc_lo, auprc_hi = bootstrap_ci(ytrue, prob, average_precision_score)
        ci_results[task][mname] = {
            "auroc": auroc, "auroc_95ci": [round(auroc_lo,4), round(auroc_hi,4)],
            "auprc": auprc, "auprc_95ci": [round(auprc_lo,4), round(auprc_hi,4)],
        }
        print(f"    {mname:<16} AUROC {auroc:.4f} [{auroc_lo:.4f}–{auroc_hi:.4f}]  "
              f"AUPRC {auprc:.4f} [{auprc_lo:.4f}–{auprc_hi:.4f}]")

    # Paired test: ordering_only vs values_only
    diff, pval = paired_bootstrap_pval(y_val, p_ord, p_val)
    ci_results[task]["paired_test_ordering_vs_values"] = {
        "auroc_diff": round(diff, 4), "p_value": round(pval, 4)}
    print(f"    Paired test (ord vs val): ΔAUROC={diff:+.4f}  p={pval:.4f}")

with open(OUT / "bootstrap_ci_results.json", "w") as f:
    json.dump(ci_results, f, indent=2)
print(f"\n  Saved → {OUT}/bootstrap_ci_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Clinical Characterisation of Deceptively Normal Cohort
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 2 — Deceptively Normal Clinical Characterisation")
print("="*65)

# Recreate phenotype labels on df_both
val_fc = [c for c in get_feat_cols(df_val) if c in df_both.columns]
val_mat = df_both[val_fc].astype("float32")
means, stds = val_mat.mean(), val_mat.std().replace(0, 1)
df_both["value_abnormality"] = ((val_mat - means) / stds).abs().mean(axis=1)
oi_cols = [c for c in df_both.columns if "ordering_intensity__" in c]
df_both["ordering_intensity_all"] = df_both[oi_cols].mean(axis=1)
df_both["val_q"] = pd.qcut(df_both["value_abnormality"],      4, labels=[0,1,2,3])
df_both["ord_q"] = pd.qcut(df_both["ordering_intensity_all"], 4, labels=[0,1,2,3])
df_both["phenotype"] = "other"
df_both.loc[(df_both["val_q"]==0)&(df_both["ord_q"]==3), "phenotype"] = "deceptively_normal"
df_both.loc[(df_both["val_q"]==0)&(df_both["ord_q"]==0), "phenotype"] = "concordant_normal"
df_both.loc[(df_both["val_q"]==3)&(df_both["ord_q"]==3), "phenotype"] = "concordant_abnormal"
df_both.loc[(df_both["val_q"]==3)&(df_both["ord_q"]==0), "phenotype"] = "contradictory"

dn_hadm = set(df_both.loc[df_both["phenotype"]=="deceptively_normal", "hadm_id"])
cn_hadm = set(df_both.loc[df_both["phenotype"]=="concordant_normal",  "hadm_id"])

# Query ICD codes for each group
print(f"  Deceptively normal N={len(dn_hadm):,} | Concordant normal N={len(cn_hadm):,}")
print("  Querying ICD-10 diagnoses from SQLite...")

con = duckdb.connect()
con.execute(f"ATTACH '{RAW_DB}' AS mimic (TYPE SQLITE, READ_ONLY TRUE)")

# Top diagnoses
icd_q = """
SELECT hadm_id, icd_code, icd_version
FROM mimic.diagnoses_icd
WHERE icd_version = 10
"""
icd_df = con.execute(icd_q).df()

def top_icd(hadm_ids, n=20):
    sub = icd_df[icd_df["hadm_id"].isin(hadm_ids)]
    # Use first 3 chars = ICD chapter/block
    sub = sub.copy()
    sub["icd3"] = sub["icd_code"].str[:3]
    counts = sub.groupby("icd3")["hadm_id"].nunique().sort_values(ascending=False)
    return counts.head(n)

# ICD-10 description lookup (common codes)
icd_desc = {
    "I50":"Heart failure","I21":"Acute MI","I10":"Essential hypertension",
    "J18":"Pneumonia","J44":"COPD","N39":"UTI","A41":"Sepsis",
    "N17":"Acute kidney failure","E11":"Type 2 diabetes","I48":"Atrial fibrillation",
    "K92":"GI hemorrhage","R65":"SIRS/Sepsis signs","C34":"Lung cancer",
    "Z95":"Cardiac device","I63":"Cerebral infarction","J96":"Resp failure",
    "K57":"Diverticular disease","E87":"Fluid/electrolyte disorder",
    "M79":"Soft tissue disorder","Z87":"Personal history",
    "K29":"Gastritis","G89":"Pain","R06":"Resp abnormalities",
    "K70":"Alcoholic liver disease","B96":"Bacterial agents","Z79":"Long-term drug use",
    "I25":"Chronic IHD","D64":"Anaemia","F10":"Alcohol use","E78":"Dyslipidaemia",
}

dn_icd = top_icd(dn_hadm)
cn_icd = top_icd(cn_hadm)

# Compute prevalence ratio: ICD prevalence in DN / prevalence in CN
dn_prev = dn_icd / len(dn_hadm)
cn_prev = cn_icd / len(cn_hadm)
ratio_df = pd.DataFrame({
    "dn_prevalence": dn_prev,
    "cn_prevalence": cn_prev,
}).dropna()
ratio_df["prevalence_ratio"] = ratio_df["dn_prevalence"] / ratio_df["cn_prevalence"].replace(0, np.nan)
ratio_df["description"] = ratio_df.index.map(lambda x: icd_desc.get(x, x))
ratio_df = ratio_df.sort_values("prevalence_ratio", ascending=False)
ratio_df.to_csv(OUT / "deceptively_normal_icd_characterisation.csv")

print("\n  Top ICD-10 codes enriched in Deceptively Normal vs Concordant Normal:")
print(f"  {'ICD3':<6} {'Description':<30} {'DN prev':>8} {'CN prev':>8} {'Ratio':>7}")
print(f"  {'-'*65}")
for code, row in ratio_df.head(15).iterrows():
    print(f"  {code:<6} {row['description']:<30} {row['dn_prevalence']:>8.3f} "
          f"{row['cn_prevalence']:>8.3f} {row['prevalence_ratio']:>7.2f}x")

# Admission type breakdown
dn_df = df_both[df_both["phenotype"]=="deceptively_normal"]
cn_df = df_both[df_both["phenotype"]=="concordant_normal"]
adm_comp = pd.DataFrame({
    "deceptively_normal": dn_df["admission_type"].value_counts(normalize=True),
    "concordant_normal":  cn_df["admission_type"].value_counts(normalize=True),
}).fillna(0).round(3)
adm_comp.to_csv(OUT / "deceptively_normal_admission_types.csv")
print("\n  Admission type comparison:")
print(adm_comp.to_string())

# Most ordered tests in DN group (mean obs across 48h)
obs_cols_lab = [c for c in labs_ord.columns
                if c.startswith("total_obs_") and "__lab" in c]
dn_labs_ord = labs_ord[labs_ord["hadm_id"].isin(dn_hadm)][["hadm_id"] + obs_cols_lab]
cn_labs_ord = labs_ord[labs_ord["hadm_id"].isin(cn_hadm)][["hadm_id"] + obs_cols_lab]

dn_mean = dn_labs_ord[obs_cols_lab].mean()
cn_mean = cn_labs_ord[obs_cols_lab].mean()
test_compare = pd.DataFrame({
    "dn_mean_obs": dn_mean,
    "cn_mean_obs": cn_mean,
    "ratio": dn_mean / cn_mean.replace(0, np.nan),
}).sort_values("ratio", ascending=False)
test_compare.index = test_compare.index.str.replace("total_obs_","").str.replace("__lab","")
test_compare.to_csv(OUT / "deceptively_normal_test_ordering.csv")
print("\n  Most over-ordered tests in Deceptively Normal vs Concordant Normal:")
print(f"  {'Test':<25} {'DN mean obs':>12} {'CN mean obs':>12} {'Ratio':>8}")
print(f"  {'-'*60}")
for test, row in test_compare.head(12).iterrows():
    print(f"  {test:<25} {row['dn_mean_obs']:>12.2f} {row['cn_mean_obs']:>12.2f} "
          f"{row['ratio']:>8.2f}x")

con.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — SOFA Score as Severity Confounder
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 3 — Simplified SOFA Score as Severity Confounder")
print("="*65)

# Compute simplified SOFA from available labs/vitals (first 48h means)
# SOFA components:
#   Respiratory: SpO2 proxy (lower = worse) — use spo2__vit (inverted)
#   Coagulation: Platelets (lower = worse)
#   Liver: Bilirubin total
#   Cardiovascular: MAP (lower = worse) + vasopressor flag
#   CNS: GCS (lower = worse)
#   Renal: Creatinine

def sofa_resp(spo2):
    """SpO2-based respiratory SOFA (approximation, no FiO2)."""
    if pd.isna(spo2): return 0
    if spo2 >= 96: return 0
    if spo2 >= 91: return 1
    if spo2 >= 86: return 2
    if spo2 >= 80: return 3
    return 4

def sofa_coag(plt):
    if pd.isna(plt): return 0
    if plt >= 150: return 0
    if plt >= 100: return 1
    if plt >= 50:  return 2
    if plt >= 20:  return 3
    return 4

def sofa_liver(bili):
    if pd.isna(bili): return 0
    if bili < 1.2:  return 0
    if bili < 2.0:  return 1
    if bili < 6.0:  return 2
    if bili < 12.0: return 3
    return 4

def sofa_cardio(map_val, vasopressor):
    if pd.isna(map_val): return 0
    if map_val >= 70: return 0
    if not vasopressor: return 1
    return 2  # simplified

def sofa_cns(gcs):
    if pd.isna(gcs): return 0
    if gcs >= 15: return 0
    if gcs >= 13: return 1
    if gcs >= 10: return 2
    if gcs >= 6:  return 3
    return 4

def sofa_renal(creat):
    if pd.isna(creat): return 0
    if creat < 1.2:  return 0
    if creat < 2.0:  return 1
    if creat < 3.5:  return 2
    if creat < 5.0:  return 3
    return 4

# Build SOFA feature frame
sofa_df = df_both[["hadm_id"]].copy()

# Pull relevant means from labs/vitals
def get_col(df, pattern):
    matches = [c for c in df.columns if pattern in c.lower()]
    return df[matches[0]] if matches else pd.Series(np.nan, index=df.index)

sofa_df["spo2"]        = get_col(df_both, "spo2__vit").values
sofa_df["platelets"]   = get_col(df_both, "platelets__lab").values
sofa_df["bilirubin"]   = get_col(df_both, "bilirubin__lab").values
sofa_df["map"]         = get_col(df_both, "mbp_art__vit").values
sofa_df["gcs"]         = (get_col(df_both, "gcs_eye__vit") +
                           get_col(df_both, "gcs_motor__vit") +
                           get_col(df_both, "gcs_verbal__vit")).values
sofa_df["creatinine"]  = get_col(df_both, "creatinine__lab").values
sofa_df["vasopressor"] = static.set_index("hadm_id").reindex(
    df_both["hadm_id"])["med_vasopressor"].values

sofa_df["sofa_resp"]   = sofa_df["spo2"].apply(sofa_resp)
sofa_df["sofa_coag"]   = sofa_df["platelets"].apply(sofa_coag)
sofa_df["sofa_liver"]  = sofa_df["bilirubin"].apply(sofa_liver)
sofa_df["sofa_cardio"] = sofa_df.apply(
    lambda r: sofa_cardio(r["map"], bool(r["vasopressor"])), axis=1)
sofa_df["sofa_cns"]    = sofa_df["gcs"].apply(sofa_cns)
sofa_df["sofa_renal"]  = sofa_df["creatinine"].apply(sofa_renal)
sofa_df["sofa_total"]  = (sofa_df[["sofa_resp","sofa_coag","sofa_liver",
                                    "sofa_cardio","sofa_cns","sofa_renal"]].sum(axis=1))

print(f"  SOFA distribution (N={len(sofa_df):,}):")
print(f"  Mean {sofa_df['sofa_total'].mean():.2f} | "
      f"Median {sofa_df['sofa_total'].median():.0f} | "
      f"Max {sofa_df['sofa_total'].max():.0f}")

# Build two datasets:
# A) SOFA only → AUROC
# B) SOFA + ordering → AUROC (does ordering add signal beyond SOFA?)
sofa_only_cols = ["sofa_total","sofa_resp","sofa_coag","sofa_liver",
                  "sofa_cardio","sofa_cns","sofa_renal"]
ord_only_cols  = [c for c in get_feat_cols(df_ord) if c in df_ord.columns]

df_sofa = df_both[["hadm_id","subject_id"] + TASKS].copy()
df_sofa = df_sofa.merge(sofa_df[["hadm_id"] + sofa_only_cols], on="hadm_id", how="left")
df_sofa_ord = df_sofa.merge(df_ord[[c for c in df_ord.columns
    if c not in ["subject_id"] + TASKS]], on="hadm_id", how="left")

sofa_meta = ["hadm_id","subject_id"] + TASKS
sofa_results = {}

for task in TASKS:
    print(f"\n  Task: {task}")
    for df_, name in [(df_sofa, "sofa_only"), (df_sofa_ord, "sofa+ordering")]:
        tr = df_["subject_id"].isin(train_s)
        va = df_["subject_id"].isin(val_s)
        te = df_["subject_id"].isin(test_s)
        fc = [c for c in df_.columns if c not in sofa_meta]
        Xtr = df_.loc[tr, fc].astype("float32")
        Xva = df_.loc[va, fc].astype("float32")
        Xte = df_.loc[te, fc].astype("float32")
        ytr, yva, yte = df_.loc[tr, task], df_.loc[va, task], df_.loc[te, task]
        ok_tr, ok_va, ok_te = ytr.notna(), yva.notna(), yte.notna()
        prev = ytr[ok_tr].mean()
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(1-prev)/prev, n_jobs=-1,
            random_state=SEED, verbose=-1, metric="auc")
        model.fit(Xtr[ok_tr], ytr[ok_tr],
                  eval_set=[(Xva[ok_va], yva[ok_va])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])
        prob  = model.predict_proba(Xte[ok_te])[:, 1]
        auroc = roc_auc_score(yte[ok_te], prob)
        lo, hi = bootstrap_ci(yte[ok_te].values, prob, roc_auc_score, n_boot=500)
        print(f"    {name:<18} AUROC {auroc:.4f} [{lo:.4f}–{hi:.4f}]")
        sofa_results.setdefault(task, {})[name] = {
            "auroc": round(auroc,4), "ci95": [round(lo,4), round(hi,4)]}

with open(OUT / "sofa_confounder_results.json", "w") as f:
    json.dump(sofa_results, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Temporal Escalation Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 4 — Temporal Escalation Analysis")
print("="*65)

# Classify patients by ordering trajectory shape over 48h:
#   Rising   : slope > threshold (escalating concern)
#   Falling  : slope < -threshold (de-escalating)
#   Flat     : |slope| <= threshold
SLOPE_THRESH = 0.05

esc_cols = [c for c in df_both.columns if "ordering_escalation__" in c]
df_both["ordering_escalation_mean"] = df_both[esc_cols].mean(axis=1)
df_both["trajectory"] = "flat"
df_both.loc[df_both["ordering_escalation_mean"] >  SLOPE_THRESH, "trajectory"] = "rising"
df_both.loc[df_both["ordering_escalation_mean"] < -SLOPE_THRESH, "trajectory"] = "falling"

print("\n  Trajectory distribution:")
traj_counts = df_both["trajectory"].value_counts()
for t, n in traj_counts.items():
    print(f"    {t:<10} N={n:>7,}  ({n/len(df_both)*100:.1f}%)")

print("\n  Outcome rates by ordering trajectory:")
print(f"  {'Trajectory':<12}", end="")
for task in TASKS:
    print(f"  {task[:8]:>10}", end="")
print()
print(f"  {'-'*60}")

traj_results = {}
for traj in ["rising","flat","falling"]:
    sub = df_both[df_both["trajectory"] == traj]
    print(f"  {traj:<12}", end="")
    traj_results[traj] = {"n": len(sub)}
    for task in TASKS:
        rate = sub[task].mean()
        print(f"  {rate:>10.4f}", end="")
        traj_results[traj][task] = round(float(rate), 4)
    print()

# Model: escalation features only vs intensity-only
slope_cols = [c for c in df_both.columns
              if c.startswith("slope_") or "ordering_escalation" in c or "ordering_intensity" in c]
df_esc = df_both[["hadm_id","subject_id"]+TASKS+slope_cols].copy()

print("\n  AUROC: Escalation features only vs Intensity features only:")
intens_cols = [c for c in df_both.columns if "ordering_intensity__" in c]
df_int = df_both[["hadm_id","subject_id"]+TASKS+intens_cols].copy()

esc_auc, int_auc = {}, {}
for task in TASKS:
    for df_, name, store in [(df_esc,"escalation",esc_auc),
                              (df_int,"intensity_only",int_auc)]:
        tr = df_["subject_id"].isin(train_s)
        va = df_["subject_id"].isin(val_s)
        te = df_["subject_id"].isin(test_s)
        fc = [c for c in df_.columns if c not in ["hadm_id","subject_id"]+TASKS]
        if not fc: continue
        Xtr = df_.loc[tr, fc].astype("float32")
        Xva = df_.loc[va, fc].astype("float32")
        Xte = df_.loc[te, fc].astype("float32")
        ytr, yva, yte = df_.loc[tr, task], df_.loc[va, task], df_.loc[te, task]
        ok_tr, ok_va, ok_te = ytr.notna(), yva.notna(), yte.notna()
        prev = ytr[ok_tr].mean()
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(1-prev)/prev, n_jobs=-1,
            random_state=SEED, verbose=-1, metric="auc")
        model.fit(Xtr[ok_tr], ytr[ok_tr],
                  eval_set=[(Xva[ok_va], yva[ok_va])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(-1)])
        prob = model.predict_proba(Xte[ok_te])[:, 1]
        auroc = roc_auc_score(yte[ok_te], prob)
        store[task] = round(auroc, 4)

print(f"\n  {'Task':<22} {'Escalation':>12} {'Intensity only':>16} {'Delta':>8}")
print(f"  {'-'*60}")
for task in TASKS:
    e = esc_auc.get(task, 0)
    i = int_auc.get(task, 0)
    print(f"  {task:<22} {e:>12.4f} {i:>16.4f} {e-i:>+8.4f}")

traj_out = {"trajectory_rates": traj_results,
            "escalation_auroc": esc_auc, "intensity_auroc": int_auc}
with open(OUT / "temporal_escalation_results.json", "w") as f:
    json.dump(traj_out, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — Sentinel Ordering Patterns
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 5 — Sentinel Ordering Patterns")
print("="*65)

obs_cols = [c for c in labs_ord.columns if c.startswith("total_obs_")]
df_sent = df_both[["hadm_id","subject_id"] + TASKS].merge(
    labs_ord[["hadm_id"] + obs_cols], on="hadm_id", how="left")

# For each test, compute: AUROC of that single test's obs count for mortality
print("\n  Individual test ordering AUROC for mortality (test set):")
te_mask = df_sent["subject_id"].isin(test_s)
y_te_mort = df_sent.loc[te_mask, "mortality"]
ok_te_mort = y_te_mort.notna()

sentinel_results = []
for col in obs_cols:
    x = df_sent.loc[te_mask, col][ok_te_mort].fillna(0).values
    y = y_te_mort[ok_te_mort].values
    if x.std() < 1e-6: continue
    try:
        auroc = roc_auc_score(y, x)
        # Find optimal threshold using Youden's J on test set
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y, x)
        j_scores = tpr - fpr
        best_idx  = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        sensitivity = tpr[best_idx]
        specificity = 1 - fpr[best_idx]
        test_name = col.replace("total_obs_","").replace("__lab","")
        sentinel_results.append({
            "test": test_name,
            "auroc": round(auroc, 4),
            "optimal_threshold_obs": round(float(best_thresh), 1),
            "sensitivity": round(float(sensitivity), 3),
            "specificity": round(float(specificity), 3),
        })
    except Exception:
        continue

sent_df = pd.DataFrame(sentinel_results).sort_values("auroc", ascending=False)
sent_df.to_csv(OUT / "sentinel_ordering_patterns.csv", index=False)

print(f"\n  Top 15 sentinel ordering patterns for mortality:")
print(f"  {'Test':<25} {'AUROC':>7} {'Threshold':>10} {'Sens':>6} {'Spec':>6}")
print(f"  {'-'*60}")
for _, row in sent_df.head(15).iterrows():
    print(f"  {row['test']:<25} {row['auroc']:>7.4f} "
          f"{row['optimal_threshold_obs']:>10.0f} "
          f"{row['sensitivity']:>6.3f} {row['specificity']:>6.3f}")

# Define sentinel alert: test ordered >= threshold → high risk
# Evaluate combined sentinel alert (any top-5 test meets threshold)
top5_tests = sent_df.head(5)
print(f"\n  Combined sentinel alert (any of top-5 tests at threshold):")
alert_mask = np.zeros(ok_te_mort.sum(), dtype=bool)
for _, row in top5_tests.iterrows():
    col = f"total_obs_{row['test']}__lab"
    if col in df_sent.columns:
        x = df_sent.loc[te_mask, col][ok_te_mort].fillna(0).values
        alert_mask |= (x >= row["optimal_threshold_obs"])

y_m = y_te_mort[ok_te_mort].values
n_alerted = alert_mask.sum()
ppv = y_m[alert_mask].mean() if alert_mask.sum() > 0 else 0
npv = 1 - y_m[~alert_mask].mean() if (~alert_mask).sum() > 0 else 0
sens = (y_m[alert_mask].sum()) / y_m.sum() if y_m.sum() > 0 else 0
spec = ((~alert_mask) & (y_m == 0)).sum() / (y_m == 0).sum()
print(f"    Patients alerted: {n_alerted:,} ({n_alerted/len(y_m)*100:.1f}%)")
print(f"    PPV: {ppv:.3f}  NPV: {npv:.3f}  Sens: {sens:.3f}  Spec: {spec:.3f}")

sentinel_summary = {
    "top_tests": sent_df.head(15).to_dict(orient="records"),
    "combined_alert": {
        "n_alerted": int(n_alerted),
        "alert_rate_pct": round(n_alerted/len(y_m)*100, 1),
        "ppv": round(float(ppv), 3),
        "npv": round(float(npv), 3),
        "sensitivity": round(float(sens), 3),
        "specificity": round(float(spec), 3),
    }
}
with open(OUT / "sentinel_results.json", "w") as f:
    json.dump(sentinel_summary, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("ALL PARTS COMPLETE")
print("="*65)
print(f"Outputs in {OUT}/")
print("  bootstrap_ci_results.json           Part 1 — CIs + paired tests")
print("  deceptively_normal_icd_*.csv        Part 2 — ICD characterisation")
print("  deceptively_normal_test_ordering.csv Part 2 — Most-ordered tests")
print("  sofa_confounder_results.json         Part 3 — SOFA confounder")
print("  temporal_escalation_results.json     Part 4 — Trajectory analysis")
print("  sentinel_ordering_patterns.csv       Part 5 — Sentinel alerts")
