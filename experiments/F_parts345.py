#!/usr/bin/env python
"""
Parts 3, 4, 5 of F_paper_improvements — resuming after Parts 1 & 2 completed.
Fixes: (a) string columns excluded from feature matrices, (b) correct get_col patterns.
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT   = Path(__file__).parents[2]
DATA   = ROOT / "data" / "processed"
RAW_DB = ROOT / "data" / "raw" / "mimic_iv_2_2.db"
OUT    = ROOT / "1_ordering_paper" / "results" / "F_paper_improvements"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED  = 42

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
cohort     = pd.read_parquet(DATA / "cohort.parquet")
labels     = pd.read_parquet(DATA / "labels.parquet")
static     = pd.read_parquet(DATA / "static.parquet")
labs_long  = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
vitals_long= pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

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
        grp  = grp.sort_values("time_bin")
        obs  = grp[obs_cols].values.astype(float)
        t    = np.arange(len(obs))
        feat = {"hadm_id": hadm_id}
        for i, col in enumerate(obs_cols):
            base = col[:-4]
            vals = obs[:, i]
            feat[f"total_obs_{base}__{name}"]  = vals.sum()
            feat[f"intensity_{base}__{name}"]  = (vals > 0).mean()
            feat[f"slope_{base}__{name}"]      = (
                np.polyfit(t, vals, 1)[0] if vals.sum() > 0 and len(t) > 1 else 0.0)
        binary = (obs > 0).astype(float)
        tpb    = binary.sum(axis=1)
        feat[f"ordering_intensity__{name}"]  = binary.mean()
        feat[f"ordering_diversity__{name}"]  = (obs.sum(axis=0) > 0).sum()
        feat[f"ordering_breadth__{name}"]    = tpb.mean()
        feat[f"ordering_escalation__{name}"] = (
            np.polyfit(t, tpb, 1)[0] if len(t) > 1 else 0.0)
        rows.append(feat)
    return pd.DataFrame(rows)

print("  Aggregating values and ordering features...")
labs_val   = agg_values(labs_long,   "lab")
vitals_val = agg_values(vitals_long, "vit")
labs_ord   = agg_ordering(labs_long,   "lab")
vitals_ord = agg_ordering(vitals_long, "vit")

# All string meta columns — never used as features
STRING_META = ["hadm_id","subject_id","race","insurance","admission_type",
               "marital_status","gender","anchor_year_group","discharge_location",
               "admission_location","deathtime","admittime","dischtime",
               "first_icu_intime"] + TASKS

base = (cohort[["hadm_id","subject_id"]].merge(labels, on=["hadm_id","subject_id"]))

def build_df(values=True, ordering=True):
    df = base.copy()
    df = df.merge(static, on="hadm_id", how="left")
    if values:   df = df.merge(labs_val,   on="hadm_id", how="left")
    if values:   df = df.merge(vitals_val, on="hadm_id", how="left")
    if ordering: df = df.merge(labs_ord,   on="hadm_id", how="left")
    if ordering: df = df.merge(vitals_ord, on="hadm_id", how="left")
    return df

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

def feat_cols(df):
    return [c for c in df.columns
            if c not in STRING_META and df[c].dtype != object]

def run_lgbm(df, task, n_est=500):
    fc = feat_cols(df)
    tr = df["subject_id"].isin(train_s)
    va = df["subject_id"].isin(val_s)
    te = df["subject_id"].isin(test_s)
    Xtr,Xva,Xte = df.loc[tr,fc].astype("float32"), df.loc[va,fc].astype("float32"), df.loc[te,fc].astype("float32")
    ytr,yva,yte = df.loc[tr,task], df.loc[va,task], df.loc[te,task]
    ok_tr,ok_va,ok_te = ytr.notna(), yva.notna(), yte.notna()
    prev  = ytr[ok_tr].mean()
    model = lgb.LGBMClassifier(
        n_estimators=n_est, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1-prev)/prev, n_jobs=-1, random_state=SEED, verbose=-1, metric="auc")
    model.fit(Xtr[ok_tr], ytr[ok_tr],
              eval_set=[(Xva[ok_va],yva[ok_va])],
              callbacks=[lgb.early_stopping(30,verbose=False), lgb.log_evaluation(-1)])
    prob  = model.predict_proba(Xte[ok_te])[:, 1]
    auroc = roc_auc_score(yte[ok_te], prob)
    lo, hi = bootstrap_ci(yte[ok_te].values, prob, roc_auc_score, n_boot=500)
    return auroc, lo, hi, prob, yte[ok_te].values

def bootstrap_ci(y_true, y_prob, fn, n_boot=500):
    rng = np.random.RandomState(SEED)
    n   = len(y_true)
    sc  = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if y_true[idx].sum() in (0, n): continue
        sc.append(fn(y_true[idx], y_prob[idx]))
    return np.percentile(sc, 2.5), np.percentile(sc, 97.5)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — SOFA Confounder
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 3 — Simplified SOFA Score as Severity Confounder")
print("="*65)

# Correct column lookups — use partial patterns matching actual names
def get_mean_col(df, keyword):
    """Find the _mean column for a given keyword (e.g. 'spo2', 'creatinine')."""
    matches = [c for c in df.columns
               if keyword.lower() in c.lower() and "_mean__" in c]
    return df[matches[0]] if matches else pd.Series(np.nan, index=df.index)

# Check available columns
print("  Checking column availability for SOFA components:")
for kw in ["spo2","platelets","bilirubin","mbp_art","gcs_eye","gcs_motor",
           "gcs_verbal","creatinine"]:
    matches = [c for c in df_both.columns if kw in c.lower() and "_mean__" in c]
    print(f"    {kw:<15}: {matches[:2]}")

sofa_df = df_both[["hadm_id","subject_id"] + TASKS].copy()
sofa_df["spo2"]       = get_mean_col(df_both, "spo2").values
sofa_df["platelets"]  = get_mean_col(df_both, "platelets").values
sofa_df["bilirubin"]  = get_mean_col(df_both, "bilirubin").values
sofa_df["map"]        = get_mean_col(df_both, "mbp_art").values
sofa_df["gcs"]        = (get_mean_col(df_both, "gcs_eye") +
                          get_mean_col(df_both, "gcs_motor") +
                          get_mean_col(df_both, "gcs_verbal")).values
sofa_df["creatinine"] = get_mean_col(df_both, "creatinine").values
sofa_df["vasopressor"]= static.set_index("hadm_id").reindex(
    df_both["hadm_id"].values)["med_vasopressor"].values

def sofa_resp(s):
    if pd.isna(s): return 0
    return 0 if s>=96 else 1 if s>=91 else 2 if s>=86 else 3 if s>=80 else 4

def sofa_coag(p):
    if pd.isna(p): return 0
    return 0 if p>=150 else 1 if p>=100 else 2 if p>=50 else 3 if p>=20 else 4

def sofa_liver(b):
    if pd.isna(b): return 0
    return 0 if b<1.2 else 1 if b<2 else 2 if b<6 else 3 if b<12 else 4

def sofa_cardio(m, v):
    if pd.isna(m): return 0
    return 0 if m>=70 else 2 if bool(v) else 1

def sofa_cns(g):
    if pd.isna(g): return 0
    return 0 if g>=15 else 1 if g>=13 else 2 if g>=10 else 3 if g>=6 else 4

def sofa_renal(c):
    if pd.isna(c): return 0
    return 0 if c<1.2 else 1 if c<2 else 2 if c<3.5 else 3 if c<5 else 4

sofa_df["sofa_resp"]   = sofa_df["spo2"].apply(sofa_resp)
sofa_df["sofa_coag"]   = sofa_df["platelets"].apply(sofa_coag)
sofa_df["sofa_liver"]  = sofa_df["bilirubin"].apply(sofa_liver)
sofa_df["sofa_cardio"] = sofa_df.apply(
    lambda r: sofa_cardio(r["map"], r["vasopressor"]), axis=1)
sofa_df["sofa_cns"]    = sofa_df["gcs"].apply(sofa_cns)
sofa_df["sofa_renal"]  = sofa_df["creatinine"].apply(sofa_renal)
sofa_components = ["sofa_resp","sofa_coag","sofa_liver","sofa_cardio","sofa_cns","sofa_renal"]
sofa_df["sofa_total"]  = sofa_df[sofa_components].sum(axis=1)

print(f"\n  SOFA distribution: mean={sofa_df['sofa_total'].mean():.2f} | "
      f"median={sofa_df['sofa_total'].median():.1f} | "
      f"max={sofa_df['sofa_total'].max():.0f}")
print(f"  Non-null spo2: {sofa_df['spo2'].notna().sum():,} | "
      f"platelets: {sofa_df['platelets'].notna().sum():,} | "
      f"creatinine: {sofa_df['creatinine'].notna().sum():,}")

# Build SOFA-only and SOFA+Ordering feature sets
sofa_feat_cols = sofa_components + ["sofa_total"]
df_sofa = sofa_df[["hadm_id","subject_id"] + TASKS + sofa_feat_cols].copy()

# SOFA + ordering: merge ordering features onto sofa_df
ord_feat_only = [c for c in feat_cols(df_ord)
                 if c not in ["hadm_id","subject_id"] + TASKS]
df_sofa_ord = df_sofa.merge(
    df_ord[["hadm_id"] + ord_feat_only], on="hadm_id", how="left")

sofa_results = {}
print("\n  Results (500-bootstrap 95% CI):")
print(f"  {'Task':<22} {'SOFA only':>22} {'SOFA+Ordering':>22} {'Delta':>8}")
print(f"  {'-'*78}")

for task in TASKS:
    a1, l1, h1, _, _ = run_lgbm(df_sofa,     task)
    a2, l2, h2, _, _ = run_lgbm(df_sofa_ord, task)
    delta = a2 - a1
    print(f"  {task:<22} {a1:.4f} [{l1:.4f}–{h1:.4f}]  "
          f"{a2:.4f} [{l2:.4f}–{h2:.4f}]  {delta:>+8.4f}")
    sofa_results[task] = {
        "sofa_only":    {"auroc": round(a1,4), "ci95": [round(l1,4), round(h1,4)]},
        "sofa+ordering":{"auroc": round(a2,4), "ci95": [round(l2,4), round(h2,4)]},
        "delta_auroc":  round(delta, 4)}

with open(OUT / "sofa_confounder_results.json", "w") as f:
    json.dump(sofa_results, f, indent=2)
print(f"  Saved → sofa_confounder_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 4 — Temporal Escalation Analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 4 — Temporal Escalation Analysis")
print("="*65)

esc_cols = [c for c in df_both.columns if "ordering_escalation__" in c]
df_both["escalation_mean"] = df_both[esc_cols].mean(axis=1)

THRESH = 0.05
df_both["trajectory"] = "flat"
df_both.loc[df_both["escalation_mean"] >  THRESH, "trajectory"] = "rising"
df_both.loc[df_both["escalation_mean"] < -THRESH, "trajectory"] = "falling"

print("\n  Trajectory distribution and outcome rates:")
print(f"  {'Trajectory':<12} {'N':>8}  ", end="")
for t in TASKS: print(f"  {t[:8]:>9}", end="")
print()
print(f"  {'-'*70}")

traj_results = {}
for traj in ["rising","flat","falling"]:
    sub = df_both[df_both["trajectory"]==traj]
    n   = len(sub)
    print(f"  {traj:<12} {n:>8,}", end="")
    traj_results[traj] = {"n": n}
    for task in TASKS:
        r = sub[task].mean()
        print(f"  {r:>9.4f}", end="")
        traj_results[traj][task] = round(float(r), 4)
    print()

# Model: escalation + intensity features vs intensity-only
slope_cols   = [c for c in df_both.columns
                if ("slope_" in c or "ordering_escalation" in c
                    or "ordering_intensity" in c or "ordering_breadth" in c)]
intens_cols  = [c for c in df_both.columns if "ordering_intensity__" in c]

df_esc = df_both[["hadm_id","subject_id"] + TASKS + slope_cols].copy()
df_int = df_both[["hadm_id","subject_id"] + TASKS + intens_cols].copy()

esc_res, int_res = {}, {}
print(f"\n  {'Task':<22} {'Escalation':>12} {'Intensity-only':>16} {'Delta':>8}")
print(f"  {'-'*62}")
for task in TASKS:
    ae, _, _, _, _ = run_lgbm(df_esc, task)
    ai, _, _, _, _ = run_lgbm(df_int, task)
    print(f"  {task:<22} {ae:>12.4f} {ai:>16.4f} {ae-ai:>+8.4f}")
    esc_res[task] = round(ae, 4)
    int_res[task] = round(ai, 4)

with open(OUT / "temporal_escalation_results.json", "w") as f:
    json.dump({"trajectory_rates": traj_results,
               "escalation_auroc": esc_res,
               "intensity_only_auroc": int_res}, f, indent=2)
print(f"  Saved → temporal_escalation_results.json")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 5 — Sentinel Ordering Patterns
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("PART 5 — Sentinel Ordering Patterns")
print("="*65)

obs_cols = [c for c in labs_ord.columns if c.startswith("total_obs_")]
df_sent  = df_both[["hadm_id","subject_id"] + TASKS].merge(
    labs_ord[["hadm_id"] + obs_cols], on="hadm_id", how="left")

te_mask    = df_sent["subject_id"].isin(test_s)
y_mort_all = df_sent.loc[te_mask, "mortality"]
ok_te      = y_mort_all.notna()

print(f"\n  Test set N={ok_te.sum():,} | Mortality rate {y_mort_all[ok_te].mean()*100:.1f}%")
print(f"\n  {'Test':<25} {'AUROC':>7} {'Threshold':>10} {'Sens':>6} {'Spec':>6} {'PPV':>6}")
print(f"  {'-'*62}")

sentinel_rows = []
y_te_mort = y_mort_all[ok_te].values

for col in sorted(obs_cols):
    x = df_sent.loc[te_mask, col][ok_te].fillna(0).values
    if x.std() < 1e-6: continue
    try:
        auroc = roc_auc_score(y_te_mort, x)
        fpr, tpr, thresh = roc_curve(y_te_mort, x)
        j      = tpr - fpr
        best   = np.argmax(j)
        bt     = float(thresh[best])
        sens   = float(tpr[best])
        spec   = float(1 - fpr[best])
        alert  = (x >= bt)
        ppv    = float(y_te_mort[alert].mean()) if alert.sum() > 0 else 0.0
        name   = col.replace("total_obs_","").replace("__lab","")
        sentinel_rows.append({
            "test": name, "auroc": round(auroc,4),
            "threshold_obs": round(bt,1),
            "sensitivity": round(sens,3),
            "specificity": round(spec,3),
            "ppv": round(ppv,3),
        })
    except Exception:
        continue

sent_df = pd.DataFrame(sentinel_rows).sort_values("auroc", ascending=False)
sent_df.to_csv(OUT / "sentinel_ordering_patterns.csv", index=False)

for _, row in sent_df.head(15).iterrows():
    print(f"  {row['test']:<25} {row['auroc']:>7.4f} "
          f"{row['threshold_obs']:>10.0f} {row['sensitivity']:>6.3f} "
          f"{row['specificity']:>6.3f} {row['ppv']:>6.3f}")

# Combined sentinel alert: any top-5 test meets its threshold
top5 = sent_df.head(5)
alert_mask = np.zeros(ok_te.sum(), dtype=bool)
for _, row in top5.iterrows():
    col = f"total_obs_{row['test']}__lab"
    if col in df_sent.columns:
        x = df_sent.loc[te_mask, col][ok_te].fillna(0).values
        alert_mask |= (x >= row["threshold_obs"])

n_alerted = int(alert_mask.sum())
ppv_c  = float(y_te_mort[alert_mask].mean()) if n_alerted > 0 else 0
npv_c  = float(1 - y_te_mort[~alert_mask].mean()) if (~alert_mask).sum() > 0 else 0
sens_c = float(y_te_mort[alert_mask].sum() / y_te_mort.sum()) if y_te_mort.sum() > 0 else 0
spec_c = float(((~alert_mask) & (y_te_mort==0)).sum() / (y_te_mort==0).sum())

print(f"\n  Combined sentinel alert (any of top-5 tests at threshold):")
print(f"    Patients alerted : {n_alerted:,} ({n_alerted/len(y_te_mort)*100:.1f}% of test set)")
print(f"    PPV  : {ppv_c:.3f}  NPV : {npv_c:.3f}")
print(f"    Sens : {sens_c:.3f}  Spec: {spec_c:.3f}")

with open(OUT / "sentinel_results.json", "w") as f:
    json.dump({
        "top_tests": sent_df.head(15).to_dict(orient="records"),
        "combined_alert": {
            "n_alerted": n_alerted,
            "alert_rate_pct": round(n_alerted/len(y_te_mort)*100, 1),
            "ppv": round(ppv_c,3), "npv": round(npv_c,3),
            "sensitivity": round(sens_c,3), "specificity": round(spec_c,3),
        }
    }, f, indent=2)
print(f"  Saved → sentinel_results.json")

print("\n" + "="*65)
print("PARTS 3–5 COMPLETE")
print("="*65)
