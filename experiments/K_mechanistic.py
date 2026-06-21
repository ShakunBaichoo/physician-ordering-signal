#!/usr/bin/env python
"""
Experiment K — Mechanistic Explanation: Why Ordering Encodes Clinical Judgment
===============================================================================
Three mechanistic claims:

  1. Ordering escalation adds significant predictive value BEYOND SOFA severity
     → shows ordering captures physician-detected signal not reducible to acuity
  2. Sentinel test-ordering patterns map to recognisable clinical syndromes
     → shows ordering patterns are clinically interpretable
  3. Among patients with "stable values" (no objective abnormality in first 8h),
     ordering escalation quartile predicts mortality monotonically
     → shows physicians sense danger BEFORE values reveal it

Outputs → 1_ordering_paper/results/K_mechanistic/
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
OUT  = ROOT / "1_ordering_paper" / "results" / "K_mechanistic"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED  = 42
N_BOOT = 1000

print("=" * 65)
print("Experiment K — Mechanistic Analysis")
print("=" * 65)


# ── 1. Load data (same as Experiment A) ───────────────────────────────────────
print("\n[1] Loading data...")
cohort     = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
labels     = pd.read_parquet(DATA / "labels.parquet")
static     = pd.read_parquet(DATA / "static.parquet")
labs_long  = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")
print(f"  Cohort: {len(cohort):,} admissions")


# ── 2. Feature extraction (identical to Experiment A) ─────────────────────────
def extract_value_features(df_long, name):
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id", "time_bin") and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()


def extract_ordering_features(df_long, name):
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    rows = []
    for hadm_id, grp in df_long.groupby("hadm_id", sort=False):
        grp = grp.sort_values("time_bin")
        obs = grp[obs_cols].values.astype(float)
        n_bins = obs.shape[0]
        t = np.arange(n_bins)
        feat = {"hadm_id": hadm_id}
        for i, col in enumerate(obs_cols):
            base = col[:-4]
            vals = obs[:, i]
            feat[f"total_obs_{base}__{name}"] = vals.sum()
            feat[f"intensity_{base}__{name}"]  = (vals > 0).mean()
            slope = np.polyfit(t, vals, 1)[0] if n_bins > 1 and vals.sum() > 0 else 0.0
            feat[f"slope_{base}__{name}"] = slope
        binary = (obs > 0).astype(float)
        tests_per_bin = binary.sum(axis=1)
        feat[f"ordering_intensity__{name}"]  = binary.mean()
        feat[f"ordering_diversity__{name}"]  = (obs.sum(axis=0) > 0).sum()
        feat[f"ordering_breadth__{name}"]    = tests_per_bin.mean()
        feat[f"ordering_escalation__{name}"] = np.polyfit(t, tests_per_bin, 1)[0] if n_bins > 1 else 0.0
        rows.append(feat)
    return pd.DataFrame(rows)


print("[2] Extracting features...")
labs_val    = extract_value_features(labs_long, "lab")
vitals_val  = extract_value_features(vitals_long, "vit")
labs_ord    = extract_ordering_features(labs_long, "lab")
vitals_ord  = extract_ordering_features(vitals_long, "vit")

base = cohort.merge(labels, on=["hadm_id", "subject_id"])

def build_df(values=True, ordering=True):
    df = base.merge(static, on="hadm_id", how="left")
    if values:
        df = df.merge(labs_val,   on="hadm_id", how="left")
        df = df.merge(vitals_val, on="hadm_id", how="left")
    if ordering:
        df = df.merge(labs_ord,   on="hadm_id", how="left")
        df = df.merge(vitals_ord, on="hadm_id", how="left")
    return df

df_val  = build_df(values=True,  ordering=False)
df_ord  = build_df(values=False, ordering=True)
df_both = build_df(values=True,  ordering=True)

META = ["hadm_id", "subject_id"] + TASKS

# Patient-level split (same SEED as all experiments)
def patient_split(df):
    pat = df.groupby("subject_id")["mortality"].max().reset_index()
    pat = pat.sample(frac=1, random_state=SEED)
    n = len(pat)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    tr_s = set(pat.iloc[:n_tr]["subject_id"])
    va_s = set(pat.iloc[n_tr:n_tr + n_va]["subject_id"])
    te_s = set(pat.iloc[n_tr + n_va:]["subject_id"])
    return (df["subject_id"].isin(tr_s),
            df["subject_id"].isin(va_s),
            df["subject_id"].isin(te_s))

tr_m, va_m, te_m = patient_split(df_both)
print(f"  Split — Train {tr_m.sum():,} | Val {va_m.sum():,} | Test {te_m.sum():,}")


# ── 3. SOFA score ──────────────────────────────────────────────────────────────
print("\n[3] Computing simplified SOFA score...")

def sofa_resp(v):
    if pd.isna(v): return 0
    if v >= 96: return 0
    if v >= 91: return 1
    if v >= 86: return 2
    if v >= 80: return 3
    return 4

def sofa_coag(v):
    if pd.isna(v): return 0
    if v >= 150: return 0
    if v >= 100: return 1
    if v >= 50:  return 2
    if v >= 20:  return 3
    return 4

def sofa_liver(v):
    if pd.isna(v): return 0
    if v < 1.2:  return 0
    if v < 2.0:  return 1
    if v < 6.0:  return 2
    if v < 12.0: return 3
    return 4

def sofa_cardio(m, vp):
    if pd.isna(m): return 0
    if m >= 70: return 0
    if not vp:  return 1
    return 2

def sofa_cns(v):
    if pd.isna(v): return 0
    if v >= 15: return 0
    if v >= 13: return 1
    if v >= 10: return 2
    if v >= 6:  return 3
    return 4

def sofa_renal(v):
    if pd.isna(v): return 0
    if v < 1.2:  return 0
    if v < 2.0:  return 1
    if v < 3.5:  return 2
    if v < 5.0:  return 3
    return 4

def get_col(df, pat):
    matches = [c for c in df.columns if pat in c.lower()]
    return df[matches[0]] if matches else pd.Series(np.nan, index=df.index)

sofa = df_both[["hadm_id"]].copy()
sofa["spo2"]       = get_col(df_both, "spo2__vit").values
sofa["platelets"]  = get_col(df_both, "platelets__lab").values
sofa["bilirubin"]  = get_col(df_both, "bilirubin__lab").values
sofa["map"]        = get_col(df_both, "mbp_art__vit").values
gcs_eye   = get_col(df_both, "gcs_eye__vit")
gcs_motor = get_col(df_both, "gcs_motor__vit")
gcs_verbal= get_col(df_both, "gcs_verbal__vit")
sofa["gcs"] = (gcs_eye + gcs_motor + gcs_verbal).values
sofa["creatinine"] = get_col(df_both, "creatinine__lab").values
vp_idx = static.set_index("hadm_id").reindex(df_both["hadm_id"].values)["med_vasopressor"].values
sofa["vasopressor"] = vp_idx

sofa["sofa_resp"]   = sofa["spo2"].apply(sofa_resp)
sofa["sofa_coag"]   = sofa["platelets"].apply(sofa_coag)
sofa["sofa_liver"]  = sofa["bilirubin"].apply(sofa_liver)
sofa["sofa_cardio"] = sofa.apply(lambda r: sofa_cardio(r["map"], bool(r["vasopressor"])), axis=1)
sofa["sofa_cns"]    = sofa["gcs"].apply(sofa_cns)
sofa["sofa_renal"]  = sofa["creatinine"].apply(sofa_renal)
sofa["sofa_total"]  = sofa[["sofa_resp","sofa_coag","sofa_liver",
                              "sofa_cardio","sofa_cns","sofa_renal"]].sum(axis=1)

SOFA_COLS = ["sofa_resp","sofa_coag","sofa_liver","sofa_cardio","sofa_cns","sofa_renal","sofa_total"]
df_sofa = df_both[["hadm_id","subject_id"] + TASKS].merge(
    sofa[["hadm_id"] + SOFA_COLS], on="hadm_id", how="left")

print(f"  SOFA mean={sofa['sofa_total'].mean():.2f}  median={sofa['sofa_total'].median():.0f}  max={sofa['sofa_total'].max():.0f}")


# ── 4. LightGBM helpers ────────────────────────────────────────────────────────
LGBM = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1,
            random_state=SEED, verbose=-1, metric="auc")

rng = np.random.default_rng(SEED)

def bootstrap_auroc(y, p):
    base = roc_auc_score(y, p)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            boot.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return round(base, 4), round(lo, 4), round(hi, 4)


def paired_bootstrap_auc_diff(y, p_a, p_b):
    """Paired bootstrap test for AUROC(model A) - AUROC(model B)."""
    base = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            boot.append(roc_auc_score(yb, p_a[idx]) - roc_auc_score(yb, p_b[idx]))
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0)) if len(boot) else np.nan
    return float(base), float(lo), float(hi), float(min(p, 1.0)) if not np.isnan(p) else np.nan

def train_predict(df, feat_cols, task, tr_mask, va_mask, te_mask):
    """Train LightGBM and return test predictions."""
    tr = df["subject_id"].isin(df_both.loc[tr_mask, "subject_id"])
    va = df["subject_id"].isin(df_both.loc[va_mask, "subject_id"])
    te = df["subject_id"].isin(df_both.loc[te_mask, "subject_id"])

    X_tr = df.loc[tr, feat_cols].astype("float32")
    X_va = df.loc[va, feat_cols].astype("float32")
    X_te = df.loc[te, feat_cols].astype("float32")
    y_tr, y_va, y_te = df.loc[tr, task], df.loc[va, task], df.loc[te, task]

    ok_tr, ok_va, ok_te = y_tr.notna(), y_va.notna(), y_te.notna()
    prev = y_tr[ok_tr].mean()
    pos_w = (1 - prev) / prev if prev > 0 else 1.0

    m = lgb.LGBMClassifier(scale_pos_weight=pos_w, **LGBM)
    m.fit(X_tr[ok_tr], y_tr[ok_tr],
          eval_set=[(X_va[ok_va], y_va[ok_va])],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(-1)])

    p = m.predict_proba(X_te[ok_te])[:, 1]
    return p, y_te[ok_te].values, m, feat_cols


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1 — Ordering escalation vs SOFA: incremental AUROC
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 1 — Ordering escalation beyond SOFA severity (mortality)")
print("=" * 65)

# Ordering escalation features only (no static, no values)
esc_cols = [c for c in df_ord.columns
            if "ordering_escalation" in c or "ordering_intensity" in c
            or "ordering_diversity" in c or "ordering_breadth" in c]

sofa_results = []
sofa_paired_rows = []
sofa_pred_rows = []

for task in ["mortality", "sepsis", "aki"]:
    print(f"\n  Task: {task}")

    # Model A: SOFA only
    sofa_feat = [c for c in SOFA_COLS if c in df_sofa.columns]
    p_sofa, y_te_sofa, _, _ = train_predict(df_sofa, sofa_feat, task, tr_m, va_m, te_m)
    a_sofa, lo_sofa, hi_sofa = bootstrap_auroc(y_te_sofa, p_sofa)
    print(f"    SOFA-only       AUROC={a_sofa:.3f} [{lo_sofa:.3f}–{hi_sofa:.3f}]")

    # Model B: ordering escalation only
    esc_df = df_both[["hadm_id","subject_id"] + TASKS].merge(
        df_ord[["hadm_id"] + [c for c in esc_cols if c in df_ord.columns]],
        on="hadm_id", how="left")
    esc_feat_cols = [c for c in esc_cols if c in esc_df.columns]
    p_esc, y_te_esc, _, _ = train_predict(esc_df, esc_feat_cols, task, tr_m, va_m, te_m)
    a_esc, lo_esc, hi_esc = bootstrap_auroc(y_te_esc, p_esc)
    print(f"    Ordering-only   AUROC={a_esc:.3f} [{lo_esc:.3f}–{hi_esc:.3f}]")

    # Model C: SOFA + ordering escalation
    combo_df = df_sofa.merge(
        df_ord[["hadm_id"] + [c for c in esc_cols if c in df_ord.columns]],
        on="hadm_id", how="left")
    combo_feat = sofa_feat + [c for c in esc_cols if c in combo_df.columns]
    p_combo, y_te_combo, _, _ = train_predict(combo_df, combo_feat, task, tr_m, va_m, te_m)
    a_combo, lo_combo, hi_combo = bootstrap_auroc(y_te_combo, p_combo)
    delta = round(a_combo - a_sofa, 4)
    print(f"    SOFA+Ordering   AUROC={a_combo:.3f} [{lo_combo:.3f}–{hi_combo:.3f}]  Δ={delta:+.3f}")

    sofa_results.append({
        "task": task,
        "sofa_only_auroc": a_sofa, "sofa_only_ci_lo": lo_sofa, "sofa_only_ci_hi": hi_sofa,
        "ordering_only_auroc": a_esc, "ordering_only_ci_lo": lo_esc, "ordering_only_ci_hi": hi_esc,
        "combined_auroc": a_combo, "combined_ci_lo": lo_combo, "combined_ci_hi": hi_combo,
        "delta_auroc": delta,
    })
    if not np.array_equal(y_te_sofa, y_te_esc) or not np.array_equal(y_te_sofa, y_te_combo):
        raise RuntimeError(f"Test labels are not aligned for SOFA paired comparison: {task}")
    preds = {
        "sofa_only": p_sofa,
        "ordering_escalation_only": p_esc,
        "sofa_plus_ordering": p_combo,
    }
    for model_name, prob in preds.items():
        sofa_pred_rows.append(pd.DataFrame({
            "task": task,
            "model": model_name,
            "row_index": np.arange(len(y_te_sofa)),
            "y_true": y_te_sofa,
            "y_prob": prob,
        }))
    for model_a, model_b in [
        ("ordering_escalation_only", "sofa_only"),
        ("sofa_plus_ordering", "sofa_only"),
        ("sofa_plus_ordering", "ordering_escalation_only"),
    ]:
        diff, lo, hi, pval = paired_bootstrap_auc_diff(y_te_sofa, preds[model_a], preds[model_b])
        sofa_paired_rows.append({
            "dataset": "MIMIC-IV test",
            "task": task,
            "model_a": model_a,
            "model_b": model_b,
            "comparison": f"{model_a} - {model_b}",
            "n": int(len(y_te_sofa)),
            "n_pos": int(y_te_sofa.sum()),
            "auc_a": float(roc_auc_score(y_te_sofa, preds[model_a])),
            "auc_b": float(roc_auc_score(y_te_sofa, preds[model_b])),
            "auc_diff": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "p_value_two_sided": pval,
        })

sofa_df_out = pd.DataFrame(sofa_results)
sofa_df_out.to_csv(OUT / "sofa_incremental_auroc.csv", index=False)
pd.DataFrame(sofa_paired_rows).to_csv(OUT / "sofa_paired_auc_comparisons.csv", index=False)
pd.concat(sofa_pred_rows, ignore_index=True).to_parquet(
    OUT / "sofa_test_predictions.parquet", index=False
)
print(f"\n  → sofa_incremental_auroc.csv")
print(f"  → sofa_paired_auc_comparisons.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2 — Sentinel ordering patterns per clinical outcome
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 2 — Sentinel test-ordering patterns (feature importance)")
print("=" * 65)

sentinel_rows = []

# Train ordering_only model for each task; extract feature importance
ord_feat_cols = [c for c in df_ord.columns if c not in META and c != "hadm_id"]
# Filter to total_obs_ features for interpretability
total_obs_cols = [c for c in ord_feat_cols if c.startswith("total_obs_")]

for task in TASKS:
    p_full, y_te_full, model, fcs = train_predict(df_ord, ord_feat_cols, task, tr_m, va_m, te_m)
    auroc, _, _ = bootstrap_auroc(y_te_full, p_full)

    # Feature importances (gain)
    imp = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=fcs
    ).sort_values(ascending=False)

    # Top 8 total_obs features
    top_obs = imp[[c for c in imp.index if c.startswith("total_obs_")]].head(8)

    print(f"\n  {task.upper()}  (AUROC={auroc:.3f})")
    for feat, gain in top_obs.items():
        # Strip suffixes for readable name
        clean = feat.replace("total_obs_","").replace("__lab","").replace("__vit","")

        # Compute OR: top quartile of this feature vs bottom quartile in test set
        te_mask = df_ord["subject_id"].isin(df_both.loc[te_m, "subject_id"])
        sub = df_ord[te_mask & df_ord[task].notna()].copy()
        if feat not in sub.columns:
            continue
        q25, q75 = sub[feat].quantile(0.25), sub[feat].quantile(0.75)
        high = sub[sub[feat] >= q75][task]
        low  = sub[sub[feat] <= q25][task]
        if high.sum() < 5 or low.sum() < 5 or (low == 0).all():
            or_val = np.nan
        else:
            # Simple OR
            p_hi = high.mean()
            p_lo = low.mean()
            or_val = round((p_hi / (1 - p_hi + 1e-9)) / (p_lo / (1 - p_lo + 1e-9)), 2) if p_lo > 0 else np.nan

        print(f"    {clean:<25}  gain={gain:.1f}  OR(Q4 vs Q1)={or_val}")
        sentinel_rows.append({
            "task": task, "feature": clean, "feature_col": feat,
            "importance_gain": round(gain, 2),
            "mean_high_q": round(high.mean(), 4) if not high.empty else np.nan,
            "mean_low_q":  round(low.mean(), 4)  if not low.empty  else np.nan,
            "odds_ratio": or_val,
        })

sentinel_df = pd.DataFrame(sentinel_rows)
sentinel_df.to_csv(OUT / "sentinel_patterns.csv", index=False)
print(f"\n  → sentinel_patterns.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3 — Ordering escalation quartiles in "stable values" patients
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 3 — Ordering escalation in stable-value patients")
print("=" * 65)

# Value abnormality score: mean |z-score| of all value features
val_feat_cols = [c for c in df_val.columns if c not in META]
X_val_all = df_both[val_feat_cols].astype("float32")
means = X_val_all.mean()
stds  = X_val_all.std().replace(0, 1)
z = ((X_val_all - means) / stds).abs()
df_both = df_both.copy()
df_both["value_abnormality"] = z.mean(axis=1)

# Overall ordering escalation (lab + vit combined slope)
esc_lab = f"ordering_escalation__lab"
esc_vit = f"ordering_escalation__vit"
df_both["ordering_escalation_overall"] = (
    df_both[esc_lab].fillna(0) + df_both[esc_vit].fillna(0)
) / 2.0

# "Stable values" = bottom half of value abnormality
stable_thresh = df_both["value_abnormality"].median()
df_stable = df_both[df_both["value_abnormality"] <= stable_thresh].copy()

print(f"  Stable-value patients (value_abnormality ≤ median): {len(df_stable):,}")

# Escalation quartiles within stable patients
df_stable["esc_q"] = pd.qcut(df_stable["ordering_escalation_overall"], 4,
                               labels=["Q1 (de-escalating)", "Q2", "Q3", "Q4 (escalating)"],
                               duplicates="drop")

esc_rows = []
print(f"\n  {'Quartile':<25}  {'N':>6}", end="")
for t in ["mortality","aki","sepsis"]:
    print(f"  {t[:10]:>10}", end="")
print()

for q_label, grp in df_stable.groupby("esc_q", observed=True):
    row = {"quartile": str(q_label), "N": len(grp)}
    print(f"  {str(q_label):<25}  {len(grp):>6}", end="")
    for t in ["mortality","aki","sepsis"]:
        rate = grp[t].mean()
        row[t] = round(rate, 4)
        print(f"  {rate:>10.4f}", end="")
    print()
    esc_rows.append(row)

esc_df_out = pd.DataFrame(esc_rows)
esc_df_out.to_csv(OUT / "escalation_mortality_by_quartile.csv", index=False)
print(f"\n  → escalation_mortality_by_quartile.csv")

# Also in the full cohort
df_both["esc_q_full"] = pd.qcut(df_both["ordering_escalation_overall"], 4,
                                  labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
full_esc_rows = []
for q_label, grp in df_both.groupby("esc_q_full", observed=True):
    row = {"quartile": str(q_label), "N": len(grp),
           "context": "all_patients"}
    for t in ["mortality","aki","sepsis"]:
        row[t] = round(grp[t].mean(), 4)
    full_esc_rows.append(row)

pd.concat([pd.DataFrame(esc_rows).assign(context="stable_values_only"),
           pd.DataFrame(full_esc_rows)]).to_csv(
    OUT / "escalation_mortality_by_quartile.csv", index=False)


# ── Summary JSON ──────────────────────────────────────────────────────────────
summary = {
    "sofa_incremental": sofa_df_out.to_dict(orient="records"),
    "escalation_in_stable_patients": esc_rows,
    "top_sentinel_per_task": {
        task: sentinel_df[sentinel_df["task"] == task].head(3).to_dict(orient="records")
        for task in TASKS
    }
}
with open(OUT / "K_mechanistic_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅  Experiment K complete.  Results → {OUT}")
