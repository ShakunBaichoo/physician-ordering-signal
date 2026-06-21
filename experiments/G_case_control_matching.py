#!/usr/bin/env python
"""
G_case_control_matching.py
==========================
1. Table 1 — baseline characteristics of all four ordering phenotype groups
   (Deceptively Normal, Concordant Normal, Concordant Abnormal, Contradictory)
   with standardised mean differences (DN vs CN) and p-values.

2. 1:1 propensity-score matched comparison:
   Deceptively Normal (DN) vs Concordant Normal (CN)
   Matching covariates: age + sex + Charlson CCI score (logistic propensity score)
   Reports: SMD before/after, outcome rates in matched cohort, McNemar/paired tests.

Outputs (results/novel/G_case_control/):
  table1_all_phenotypes.csv    — full Table 1
  matching_balance.csv         — SMD before / after matching
  matched_outcomes.csv         — outcome rates DN vs CN in matched cohort
  matched_cohort.csv           — hadm_ids of matched pairs
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "1_ordering_paper" / "results" / "G_case_control"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED  = 42


# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
cohort      = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
labels      = pd.read_parquet(DATA / "labels.parquet")
static      = pd.read_parquet(DATA / "static.parquet")
labs_long   = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

print(f"  Cohort: {len(cohort):,} admissions | Static: {len(static):,}")


# ── 2. Compute value features (mean across 48h per admission) ──────────────────
print("Computing value features...")

def agg_mean(df_long, suffix):
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id", "time_bin") and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{suffix}" for c in agg.columns]
    return agg.reset_index()

labs_val   = agg_mean(labs_long,   "lab")
vitals_val = agg_mean(vitals_long, "vit")
val_cols   = [c for c in labs_val.columns if c != "hadm_id"] + \
             [c for c in vitals_val.columns if c != "hadm_id"]


# ── 3. Compute ordering intensity (fraction of (test × bin) pairs ordered) ─────
print("Computing ordering intensity...")

def ordering_intensity(df_long):
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    oi = df_long.groupby("hadm_id", sort=False)[obs_cols].apply(
        lambda g: pd.Series({"ordering_intensity": (g.values > 0).mean()})
    ).reset_index()
    return oi

labs_oi   = ordering_intensity(labs_long)
vitals_oi = ordering_intensity(vitals_long)
# Combined ordering intensity = mean across both
oi_df = labs_oi.merge(vitals_oi, on="hadm_id", suffixes=("_lab", "_vit"))
oi_df["ordering_intensity"] = oi_df[["ordering_intensity_lab",
                                      "ordering_intensity_vit"]].mean(axis=1)
oi_df = oi_df[["hadm_id", "ordering_intensity"]]


# ── 4. Build master dataframe ──────────────────────────────────────────────────
print("Building master dataframe...")
base = cohort.merge(labels, on=["hadm_id", "subject_id"])
df = (base
      .merge(static, on="hadm_id", how="left")
      .merge(labs_val, on="hadm_id", how="left")
      .merge(vitals_val, on="hadm_id", how="left")
      .merge(oi_df, on="hadm_id", how="left"))

print(f"  Master df: {len(df):,} rows, {len(df.columns):,} columns")


# ── 5. Patient-level train/val/test split (same seed as all experiments) ────────
pat = df.groupby("subject_id")["mortality"].max().reset_index()
pat = pat.sample(frac=1, random_state=SEED)
n   = len(pat)
n_tr, n_va = int(0.70 * n), int(0.15 * n)
train_s = set(pat.iloc[:n_tr]["subject_id"])

tr_mask = df["subject_id"].isin(train_s)
print(f"  Training admissions: {tr_mask.sum():,}")


# ── 6. Z-score value features on training set → compute val_abnormality ─────────
print("Computing value abnormality (mean |z-score| of lab/vital values)...")
val_mat = df[val_cols].astype("float32")

tr_mean = val_mat.loc[tr_mask].mean()
tr_std  = val_mat.loc[tr_mask].std().replace(0, np.nan)

z_mat = (val_mat - tr_mean) / tr_std       # z-score using training stats
# NaN remains NaN (test not measured) → skipna mean so missing tests don't dilute score
df["val_abnormality"] = z_mat.abs().mean(axis=1, skipna=True).fillna(0)


# ── 7. Quartile both dimensions on training set → assign phenotypes ─────────────
print("Assigning phenotype quartiles...")

# Compute quartile boundaries on training set
oi_q_cuts  = df.loc[tr_mask, "ordering_intensity"].quantile([0.25, 0.50, 0.75]).values
val_q_cuts = df.loc[tr_mask, "val_abnormality"].quantile([0.25, 0.50, 0.75]).values

def assign_quartile(series, cuts):
    q = pd.Series(0, index=series.index, dtype=int)
    q[series > cuts[0]] = 1
    q[series > cuts[1]] = 2
    q[series > cuts[2]] = 3
    return q + 1   # 1-indexed

df["ord_q"] = assign_quartile(df["ordering_intensity"], oi_q_cuts)
df["val_q"] = assign_quartile(df["val_abnormality"], val_q_cuts)

# Phenotype labels
phenotype_map = {
    (1, 4): "Deceptively Normal (DN)",        # low values, high ordering
    (1, 1): "Concordant Normal (CN)",          # low values, low ordering
    (4, 4): "Concordant Abnormal (CA)",        # high values, high ordering
    (4, 1): "Contradictory (CT)",              # high values, low ordering
}
df["phenotype"] = df.apply(
    lambda r: phenotype_map.get((r["val_q"], r["ord_q"]), "Other"), axis=1)

ph_counts = df["phenotype"].value_counts()
print("\n  Phenotype distribution:")
for name, n in ph_counts.items():
    print(f"    {name}: {n:,} ({n/len(df)*100:.1f}%)")


# ── 8. Table 1 ─────────────────────────────────────────────────────────────────
print("\nBuilding Table 1...")

PHENOTYPES_ORDERED = [
    "Deceptively Normal (DN)",
    "Concordant Normal (CN)",
    "Concordant Abnormal (CA)",
    "Contradictory (CT)",
]

# Only the 4 main groups
df4 = df[df["phenotype"].isin(PHENOTYPES_ORDERED)].copy()

# Pre-compute emergency_admit on df4 before any slicing
emer_cols_all = [c for c in df4.columns if "admission_type_" in c and "EMER" in c]
if emer_cols_all:
    df4["emergency_admit"] = df4[emer_cols_all].max(axis=1)
else:
    df4["emergency_admit"] = 0

# Phenotype slices — defined AFTER adding derived columns
dn_df = df4[df4["phenotype"] == "Deceptively Normal (DN)"].copy()
cn_df = df4[df4["phenotype"] == "Concordant Normal (CN)"].copy()

def smd_cont(s1, s2):
    """Standardised mean difference for continuous variable."""
    m1, m2 = s1.mean(), s2.mean()
    v1, v2 = s1.var(), s2.var()
    pooled = np.sqrt((v1 + v2) / 2)
    return (m1 - m2) / pooled if pooled > 0 else 0.0

def smd_bin(p1, p2):
    """SMD for binary proportion."""
    denom = np.sqrt((p1*(1-p1) + p2*(1-p2)) / 2)
    return (p1 - p2) / denom if denom > 0 else 0.0

def fmt_cont(col, group):
    vals = df4.loc[df4["phenotype"]==group, col].dropna()
    return f"{vals.mean():.1f} ± {vals.std():.1f}" if len(vals) > 0 else "—"

def fmt_bin(col, group, total):
    vals = df4.loc[df4["phenotype"]==group, col]
    n = int(vals.sum())
    pct = 100 * n / total if total > 0 else 0
    return f"{n:,} ({pct:.1f}%)"

def kw_p(col):
    groups = [df4.loc[df4["phenotype"]==ph, col].dropna().values
              for ph in PHENOTYPES_ORDERED]
    try:
        _, p = stats.kruskal(*groups)
        return f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    except Exception:
        return "—"

def chi2_p(col):
    ct = pd.crosstab(df4["phenotype"], df4[col])
    try:
        _, p, _, _ = stats.chi2_contingency(ct)
        return f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    except Exception:
        return "—"

# Build rows
rows = []
group_ns = {ph: (df4["phenotype"]==ph).sum() for ph in PHENOTYPES_ORDERED}

def add_cont(label, col, pval_fn=kw_p):
    row = {"Variable": label, "p-value (all groups)": pval_fn(col)}
    for ph in PHENOTYPES_ORDERED:
        row[ph] = fmt_cont(col, ph)
    dn_v = dn_df[col].dropna()
    cn_v = cn_df[col].dropna()
    row["SMD (DN vs CN)"] = f"{smd_cont(dn_v, cn_v):.3f}"
    _, p = stats.mannwhitneyu(dn_v, cn_v, alternative="two-sided") if (len(dn_v)>0 and len(cn_v)>0) else (None, None)
    row["p (DN vs CN)"] = (f"{p:.4f}" if p and p >= 0.0001 else "<0.0001") if p else "—"
    rows.append(row)

def add_bin(label, col, pval_fn=None):
    row = {"Variable": label, "p-value (all groups)": ""}
    for ph in PHENOTYPES_ORDERED:
        row[ph] = fmt_bin(col, ph, group_ns[ph])
    dn_p = dn_df[col].mean()
    cn_p = cn_df[col].mean()
    row["SMD (DN vs CN)"] = f"{smd_bin(dn_p, cn_p):.3f}"
    # Chi-squared DN vs CN
    ct = pd.crosstab(
        df4.loc[df4["phenotype"].isin(["Deceptively Normal (DN)","Concordant Normal (CN)"]), "phenotype"],
        df4.loc[df4["phenotype"].isin(["Deceptively Normal (DN)","Concordant Normal (CN)"]), col])
    try:
        _, p, _, _ = stats.chi2_contingency(ct)
        row["p (DN vs CN)"] = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
    except Exception:
        row["p (DN vs CN)"] = "—"
    row["p-value (all groups)"] = chi2_p(col)
    rows.append(row)

def add_header(label):
    row = {"Variable": f"── {label} ──"}
    for ph in PHENOTYPES_ORDERED: row[ph] = ""
    row["SMD (DN vs CN)"] = ""
    row["p (DN vs CN)"] = ""
    row["p-value (all groups)"] = ""
    rows.append(row)

# N row
row = {"Variable": "N admissions"}
for ph in PHENOTYPES_ORDERED:
    row[ph] = f"{group_ns[ph]:,}"
row["SMD (DN vs CN)"] = ""
row["p (DN vs CN)"] = ""
row["p-value (all groups)"] = ""
rows.append(row)

add_header("Demographics")
add_cont("Age, years — mean (SD)", "age")
add_bin("Female sex", "is_female")

add_header("Race")
for col, label in [("race_group_white","White"), ("race_group_black","Black"),
                   ("race_group_hispanic","Hispanic"), ("race_group_asian","Asian"),
                   ("race_group_other","Other"), ("race_group_unknown","Unknown")]:
    if col in df4.columns:
        add_bin(f"  {label}", col)

add_header("Insurance")
for col, label in [("insurance_Medicare","Medicare"),("insurance_Medicaid","Medicaid"),
                   ("insurance_Other","Other")]:
    if col in df4.columns:
        add_bin(f"  {label}", col)

add_header("Comorbidities")
add_cont("Charlson CCI — mean (SD)", "cci_score")
for col, label in [
    ("chf","Congestive heart failure"),
    ("copd","COPD"),
    ("diabetes_unc","Diabetes (uncomplicated)"),
    ("diabetes_cc","Diabetes (with complications)"),
    ("renal","Renal disease"),
    ("malignancy","Malignancy"),
    ("metastatic","Metastatic cancer"),
    ("mild_liver","Mild liver disease"),
    ("severe_liver","Severe liver disease"),
    ("stroke","Stroke / hemiplegia"),
    ("dementia","Dementia"),
    ("mi","Myocardial infarction"),
    ("pvd","Peripheral vascular disease"),
    ("aids","AIDS / HIV"),
]:
    if col in df4.columns:
        add_bin(f"  {label}", col)

add_header("Admission context")
add_bin("Emergency admission", "emergency_admit")
add_bin("Prior admission", "has_prior_admission")
add_cont("Prior admissions count — mean (SD)", "n_prior_admissions")

add_header("Medications at admission")
for col in sorted([c for c in df4.columns if c.startswith("med_")]):
    label = col.replace("med_", "").replace("_", " ").title()
    add_bin(f"  {label}", col)

add_header("Outcomes")
for task, label in [("mortality","In-hospital mortality"),
                    ("readmission_30d","30-day readmission"),
                    ("aki","Acute kidney injury (AKI)"),
                    ("sepsis","Sepsis")]:
    add_bin(label, task)

table1 = pd.DataFrame(rows)
cols_order = ["Variable"] + PHENOTYPES_ORDERED + ["SMD (DN vs CN)", "p (DN vs CN)", "p-value (all groups)"]
table1 = table1[cols_order]
table1.to_csv(OUT / "table1_all_phenotypes.csv", index=False)
print(f"  Table 1 saved ({len(table1)} rows)")

# Print key rows
print("\n  ── TABLE 1 SUMMARY (key rows) ──")
key_vars = ["N admissions", "Age, years — mean (SD)", "Female sex",
            "Charlson CCI — mean (SD)", "  Congestive heart failure",
            "  Renal disease", "Emergency admission",
            "In-hospital mortality", "30-day readmission",
            "Acute kidney injury (AKI)", "Sepsis"]
t1_key = table1[table1["Variable"].isin(key_vars)]
print(t1_key.to_string(index=False))


# ── 9. Propensity Score Matching (DN vs CN) ────────────────────────────────────
print("\n" + "="*65)
print("PROPENSITY SCORE MATCHING — Deceptively Normal vs Concordant Normal")
print("="*65)

match_df = df4[df4["phenotype"].isin(
    ["Deceptively Normal (DN)", "Concordant Normal (CN)"])].copy()
match_df["is_DN"] = (match_df["phenotype"] == "Deceptively Normal (DN)").astype(int)

print(f"  Pre-match: DN={match_df['is_DN'].sum():,} | CN={(match_df['is_DN']==0).sum():,}")

# Matching covariates
cov_cols = ["age", "is_female", "cci_score"]
X = match_df[cov_cols].fillna(0).astype(float)

# Propensity score
scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)
lr     = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
lr.fit(X_sc, match_df["is_DN"])
match_df["ps"] = lr.predict_proba(X_sc)[:, 1]

print(f"  Propensity score: DN mean={match_df.loc[match_df['is_DN']==1,'ps'].mean():.3f} "
      f"| CN mean={match_df.loc[match_df['is_DN']==0,'ps'].mean():.3f}")

# 1:1 greedy nearest-neighbour matching (without replacement, caliper = 0.05)
CALIPER = 0.20   # 0.2 × pooled SD of propensity score — standard recommendation
dn_idx = match_df.index[match_df["is_DN"] == 1].tolist()
cn_idx = match_df.index[match_df["is_DN"] == 0].tolist()

dn_ps = match_df.loc[dn_idx, "ps"].values.reshape(-1, 1)
cn_ps = match_df.loc[cn_idx, "ps"].values.reshape(-1, 1)

nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
nn.fit(cn_ps)
distances, indices = nn.kneighbors(dn_ps)

matched_pairs = []
used_cn = set()
for i, (dist, cn_pos) in enumerate(zip(distances[:, 0], indices[:, 0])):
    if dist > CALIPER:
        continue
    cn_i = cn_idx[cn_pos]
    if cn_i in used_cn:
        continue
    matched_pairs.append((dn_idx[i], cn_i))
    used_cn.add(cn_i)

n_matched = len(matched_pairs)
print(f"  After 1:1 matching (caliper={CALIPER}): {n_matched:,} pairs "
      f"({n_matched/match_df['is_DN'].sum()*100:.1f}% of DN matched)")

dn_matched_idx = [p[0] for p in matched_pairs]
cn_matched_idx = [p[1] for p in matched_pairs]
matched_all    = dn_matched_idx + cn_matched_idx
matched_df     = match_df.loc[matched_all].copy()

# Save matched pairs
pairs_df = pd.DataFrame({
    "dn_hadm_id": match_df.loc[dn_matched_idx, "hadm_id"].values,
    "cn_hadm_id": match_df.loc[cn_matched_idx, "hadm_id"].values,
    "dn_ps":      match_df.loc[dn_matched_idx, "ps"].values,
    "cn_ps":      match_df.loc[cn_matched_idx, "ps"].values,
})
pairs_df.to_csv(OUT / "matched_cohort.csv", index=False)


# ── 10. Balance check: SMD before and after ────────────────────────────────────
print("\n  Balance (Standardised Mean Difference before → after matching):")
print(f"  {'Variable':<30} {'SMD Before':>12} {'SMD After':>12} {'Balanced?':>10}")
print(f"  {'-'*68}")

balance_rows = []
for col, label in [("age", "Age"), ("is_female", "Female sex"),
                   ("cci_score", "Charlson CCI"),
                   ("chf", "CHF"), ("renal", "Renal disease"),
                   ("malignancy", "Malignancy"), ("diabetes_cc", "Diabetes (cc)"),
                   ("copd", "COPD"), ("n_prior_admissions", "Prior admissions")]:
    if col not in match_df.columns:
        continue
    dn_pre  = match_df.loc[match_df["is_DN"]==1, col].dropna()
    cn_pre  = match_df.loc[match_df["is_DN"]==0, col].dropna()
    dn_post = matched_df.loc[matched_df["is_DN"]==1, col].dropna()
    cn_post = matched_df.loc[matched_df["is_DN"]==0, col].dropna()

    if col == "is_female" or df4[col].nunique() == 2:
        smd_pre  = smd_bin(dn_pre.mean(),  cn_pre.mean())
        smd_post = smd_bin(dn_post.mean(), cn_post.mean())
    else:
        smd_pre  = smd_cont(dn_pre,  cn_pre)
        smd_post = smd_cont(dn_post, cn_post)

    balanced = abs(smd_post) < 0.1
    print(f"  {label:<30} {smd_pre:>+12.3f} {smd_post:>+12.3f} {'✓' if balanced else '✗ IMBALANCED':>10}")
    balance_rows.append({
        "variable": label, "smd_before": round(smd_pre, 4),
        "smd_after": round(smd_post, 4), "balanced": bool(balanced)
    })

pd.DataFrame(balance_rows).to_csv(OUT / "matching_balance.csv", index=False)


# ── 11. Outcome rates in matched cohort ────────────────────────────────────────
print("\n  Outcome rates — matched cohort (DN vs CN):")
print(f"  {'Outcome':<28} {'DN (N=' + str(n_matched) + ')':>16} "
      f"{'CN (N=' + str(n_matched) + ')':>16} {'OR (95% CI)':>20} {'p-value':>10}")
print(f"  {'-'*94}")

outcome_rows = []
for task, label in [("mortality", "In-hospital mortality"),
                    ("readmission_30d", "30-day readmission"),
                    ("aki", "AKI"),
                    ("sepsis", "Sepsis")]:
    dn_out = matched_df.loc[matched_df["is_DN"]==1, task].dropna()
    cn_out = matched_df.loc[matched_df["is_DN"]==0, task].dropna()

    n_dn, n_cn = len(dn_out), len(cn_out)
    r_dn = dn_out.mean()
    r_cn = cn_out.mean()

    # Odds ratio + 95% CI (Woolf method)
    a, b = dn_out.sum(), n_dn - dn_out.sum()
    c, d = cn_out.sum(), n_cn - cn_out.sum()
    OR = (a*d) / (b*c) if (b*c) > 0 else np.nan
    if not np.isnan(OR) and OR > 0:
        log_OR  = np.log(OR)
        se_log  = np.sqrt(1/a + 1/b + 1/c + 1/d) if all(x > 0 for x in [a,b,c,d]) else np.nan
        if se_log and not np.isnan(se_log):
            lo_OR = np.exp(log_OR - 1.96*se_log)
            hi_OR = np.exp(log_OR + 1.96*se_log)
            or_str = f"{OR:.2f} ({lo_OR:.2f}–{hi_OR:.2f})"
        else:
            or_str = f"{OR:.2f} (CI n/a)"
    else:
        or_str = "—"

    # McNemar test on paired data (DN and CN from same matched pair)
    dn_vec = matched_df.loc[dn_matched_idx, task].values
    cn_vec = matched_df.loc[cn_matched_idx, task].values
    ok     = ~(np.isnan(dn_vec) | np.isnan(cn_vec))
    dn_v, cn_v = dn_vec[ok].astype(int), cn_vec[ok].astype(int)
    b_count = int(((dn_v==1) & (cn_v==0)).sum())   # DN=1, CN=0
    c_count = int(((dn_v==0) & (cn_v==1)).sum())   # DN=0, CN=1
    if b_count + c_count > 0:
        mcn_stat = (abs(b_count - c_count) - 1)**2 / (b_count + c_count)
        p_mcn = stats.chi2.sf(mcn_stat, df=1)
        p_str = f"{p_mcn:.4f}" if p_mcn >= 0.0001 else "<0.0001"
    else:
        p_str = "—"

    print(f"  {label:<28} {r_dn*100:>6.2f}% ({int(a):,}/{n_dn:,})"
          f"  {r_cn*100:>6.2f}% ({int(c):,}/{n_cn:,})"
          f"  {or_str:>20}  {p_str:>10}")

    outcome_rows.append({
        "outcome": label, "task": task,
        "dn_n": n_dn, "cn_n": n_cn,
        "dn_rate": round(r_dn, 4), "cn_rate": round(r_cn, 4),
        "odds_ratio": round(OR, 3) if not np.isnan(OR) else None,
        "p_mcnemar": p_str,
    })

pd.DataFrame(outcome_rows).to_csv(OUT / "matched_outcomes.csv", index=False)


# ── 12. Unmatched (full) vs matched comparison ─────────────────────────────────
print("\n  Mortality rate comparison:")
print(f"    Unmatched — DN: {dn_df['mortality'].mean()*100:.2f}% | CN: {cn_df['mortality'].mean()*100:.2f}%")
print(f"    Matched   — DN: {matched_df.loc[matched_df['is_DN']==1,'mortality'].mean()*100:.2f}% "
      f"| CN: {matched_df.loc[matched_df['is_DN']==0,'mortality'].mean()*100:.2f}%")


# ── 13. Save summary JSON ──────────────────────────────────────────────────────
summary = {
    "phenotype_counts": {ph: int(group_ns[ph]) for ph in PHENOTYPES_ORDERED},
    "matching": {
        "method": "1:1 propensity-score nearest-neighbour (caliper=0.05)",
        "covariates": cov_cols,
        "dn_pre_match": int(match_df["is_DN"].sum()),
        "cn_pre_match": int((match_df["is_DN"]==0).sum()),
        "matched_pairs": n_matched,
        "match_rate_pct": round(n_matched / match_df["is_DN"].sum() * 100, 1),
    },
    "matched_outcomes": outcome_rows,
    "balance": balance_rows,
}
with open(OUT / "matching_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*65}")
print("COMPLETE — Results saved to results/novel/G_case_control/")
print("  table1_all_phenotypes.csv  — full Table 1 with SMDs")
print("  matching_balance.csv       — SMD before/after matching")
print("  matched_outcomes.csv       — outcome rates in matched cohort")
print("  matched_cohort.csv         — matched hadm_id pairs")
print("  matching_summary.json      — full summary")
