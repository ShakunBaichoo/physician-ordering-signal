#!/usr/bin/env python
"""
P_ed_triage_validation.py
==========================
Tests the ordering signal in its most clinically relevant setting:
the Emergency Department, BEFORE the patient is admitted as an inpatient.

Research question:
  Using only ED ordering behavior (no lab values) — from the moment a
  patient arrives in the ED until they are admitted — can we predict
  inpatient outcomes (mortality, AKI, sepsis, 30d readmission)?

This is the TRUE triage scenario:
  - Physician sees patient
  - Places initial orders (lab panel, imaging, medications)
  - No results have returned yet
  - Can we predict what will happen during the inpatient stay?

Cohort: 138,733 ED stays linked to inpatient admissions in our cohort.

Features:
  (1) triage_vitals    — HR, RR, O2sat, SBP, DBP, temp, pain, acuity (from triage table)
  (2) ed_lab_ordering  — counts by test category + key sentinel tests + STAT ratio
  (3) ed_imaging       — xray, CT, MRI, echo, ECG ordered in ED
  (4) ed_full_behav    — all of the above combined (no values)

Key comparison:
  ED ordering-only vs inpatient ordering-only (0.866 mortality)
  ED ordering-only vs triage window L at t=0-4h (0.810 mortality)

Outputs → 1_ordering_paper/results/P_ed_triage/
  ed_triage_results.json
  ed_triage_auroc.csv
  fig15_ed_triage.png
"""

import json
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
DB   = ROOT / "data" / "raw" / "mimic_iv_2_2.db"
OUT  = ROOT / "1_ordering_paper" / "results" / "P_ed_triage"
FIG  = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED   = 42
N_BOOT = 1000

# Lab categories to use as ordering features
LAB_CATEGORIES = [
    "Chemistry", "Hematology", "Blood Gas", "Coagulation",
    "Microbiology", "Urinalysis", "Other",
]

# High-acuity sentinel lab tests (itemids from d_labitems)
# These encode specific clinical hypotheses (sepsis, AKI, cardiac, PE)
SENTINEL_LABS = {
    "lactate":        [50813],          # sepsis / shock
    "troponin":       [51002, 51003],   # cardiac
    "bnp":            [50963],          # heart failure
    "dimer":          [51214],          # PE / coagulopathy
    "procalcitonin":  [50889],          # sepsis
    "blood_culture":  [90220, 90221],   # sepsis (microbiology)
    "creatinine":     [50912],          # AKI
    "urine_culture":  [90232],          # UTI/sepsis
    "abg":            [50800, 50802],   # respiratory failure
}

# Imaging category mapping from poe order_subtype
IMAGING_CATS = {
    "ed_xray":     ["General Xray"],
    "ed_ct":       ["CT Scan"],
    "ed_mri":      ["MRI"],
    "ed_us":       ["Ultrasound", "Noninvasive Vascular"],
    "ed_ecg":      ["ECG"],
    "ed_echo":     ["Echo", "Stress Echo"],
    "ed_other_img":["Nuclear Med", "Angio", "Interventional Radiology",
                    "Cross-Sectional Interventional Radiology", "Interventional Neuro"],
}

# ── 1. Load cohort ─────────────────────────────────────────────────────────────
print("Loading cohort, labels, static...")
cohort = pd.read_parquet(DATA / "cohort.parquet")[
    ["hadm_id", "subject_id", "anchor_year_group"]
]
labels = pd.read_parquet(DATA / "labels.parquet")
static = pd.read_parquet(DATA / "static.parquet")
META   = ["hadm_id", "subject_id"] + TASKS


# ── 2. Query ED data from SQLite ───────────────────────────────────────────────
print("\nConnecting to MIMIC SQLite...")
con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS mimic (TYPE SQLITE, READ_ONLY TRUE)")

cohort_hids = cohort["hadm_id"].tolist()
con.execute(
    "CREATE TEMP TABLE cohort_hids AS SELECT unnest(?::BIGINT[]) AS hadm_id",
    [cohort_hids]
)

# ── 2a. Triage vitals ─────────────────────────────────────────────────────────
print("Querying triage vitals...")
triage_df = con.execute("""
    SELECT
        CAST(e.hadm_id AS BIGINT) AS hadm_id,
        t.temperature,
        t.heartrate,
        t.resprate,
        t.o2sat,
        t.sbp,
        t.dbp,
        CASE WHEN t.pain ~ '^[0-9]+$' THEN CAST(t.pain AS FLOAT) ELSE NULL END AS pain,
        t.acuity,
        (EPOCH(TRY_CAST(e.outtime AS TIMESTAMP)) -
         EPOCH(TRY_CAST(e.intime  AS TIMESTAMP))) / 3600.0  AS ed_los_hours,
        e.arrival_transport
    FROM mimic.edstays e
    LEFT JOIN mimic.triage t ON e.stay_id = t.stay_id
    JOIN cohort_hids c ON CAST(e.hadm_id AS BIGINT) = c.hadm_id
    WHERE e.disposition = 'ADMITTED'
""").df()
print(f"  Triage rows: {len(triage_df):,}")

# Encode arrival_transport as binary flags
transport_dummies = pd.get_dummies(triage_df["arrival_transport"],
                                   prefix="transport", dummy_na=True)
triage_df = pd.concat([triage_df.drop(columns=["arrival_transport"]),
                        transport_dummies], axis=1)

# For multiple ED stays per hadm_id (rare), keep last (closest to admission)
triage_df = triage_df.sort_values("hadm_id").drop_duplicates("hadm_id", keep="last")
print(f"  After dedup: {len(triage_df):,} admissions with triage data")


# ── 2b. Lab ordering in ED (from labevents) ───────────────────────────────────
print("Querying ED lab ordering (labevents in ED window)...")
ed_labs_df = con.execute("""
    SELECT
        CAST(e.hadm_id AS BIGINT) AS hadm_id,
        l.itemid,
        d.category                AS lab_category,
        l.priority,
        l.valuenum
    FROM mimic.edstays e
    JOIN mimic.labevents l ON l.hadm_id = CAST(e.hadm_id AS BIGINT)
    JOIN mimic.d_labitems d ON l.itemid = d.itemid
    JOIN cohort_hids c ON CAST(e.hadm_id AS BIGINT) = c.hadm_id
    WHERE e.disposition = 'ADMITTED'
      AND TRY_CAST(l.charttime AS TIMESTAMP) BETWEEN
          TRY_CAST(e.intime AS TIMESTAMP) AND TRY_CAST(e.outtime AS TIMESTAMP)
""").df()
print(f"  ED lab rows: {len(ed_labs_df):,}")


def build_lab_ordering_features(df):
    """
    Build per-admission lab ordering features from ED labevents:
      - Total tests, distinct tests, STAT count, STAT ratio
      - Count per lab category
      - Flag for each sentinel test (ordered yes/no + count)
    """
    rows = {}

    # Aggregate per hadm_id
    for hadm_id, grp in df.groupby("hadm_id"):
        row = {
            "hadm_id": hadm_id,
            "ed_total_labs":     len(grp),
            "ed_distinct_labs":  grp["itemid"].nunique(),
            "ed_stat_count":     (grp["priority"] == "STAT").sum(),
            "ed_stat_ratio":     (grp["priority"] == "STAT").mean(),
        }
        # Per-category counts
        for cat in LAB_CATEGORIES:
            row[f"ed_lab_{cat.lower().replace(' ', '_')}"] = (
                grp["lab_category"].str.lower().str.replace(" ", "_") ==
                cat.lower().replace(" ", "_")
            ).sum()

        # Sentinel test flags
        itemids = set(grp["itemid"].tolist())
        for name, iids in SENTINEL_LABS.items():
            row[f"ed_sentinel_{name}"] = int(bool(itemids & set(iids)))
            row[f"ed_sentinel_{name}_n"] = sum(
                (grp["itemid"] == iid).sum() for iid in iids
            )
        rows[hadm_id] = row

    feat_df = pd.DataFrame(list(rows.values()))
    return feat_df


print("  Building lab ordering features...")
lab_ord_feat = build_lab_ordering_features(ed_labs_df)
print(f"  Lab ordering features: {lab_ord_feat.shape}")


# ── 2c. Imaging ordering in ED (from poe) ─────────────────────────────────────
print("Querying ED imaging ordering...")
ed_img_df = con.execute("""
    SELECT
        CAST(e.hadm_id AS BIGINT) AS hadm_id,
        p.order_type,
        p.order_subtype
    FROM mimic.edstays e
    JOIN mimic.poe p ON CAST(p.hadm_id AS BIGINT) = CAST(e.hadm_id AS BIGINT)
    JOIN cohort_hids c ON CAST(e.hadm_id AS BIGINT) = c.hadm_id
    WHERE e.disposition = 'ADMITTED'
      AND p.order_type IN ('Radiology', 'Cardiology')
      AND TRY_CAST(p.ordertime AS TIMESTAMP) BETWEEN
          TRY_CAST(e.intime AS TIMESTAMP) AND TRY_CAST(e.outtime AS TIMESTAMP)
""").df()
print(f"  ED imaging rows: {len(ed_img_df):,}")

# Build subtype → category mapping
sub2cat = {}
for cat, subtypes in IMAGING_CATS.items():
    for st in subtypes:
        sub2cat[st] = cat

ed_img_df["img_cat"] = ed_img_df["order_subtype"].map(sub2cat)

img_rows = {}
for hadm_id, grp in ed_img_df.groupby("hadm_id"):
    row = {"hadm_id": hadm_id, "ed_total_imaging": len(grp)}
    for cat in IMAGING_CATS:
        row[cat] = (grp["img_cat"] == cat).sum()
    img_rows[hadm_id] = row

img_feat = pd.DataFrame(list(img_rows.values())) if img_rows else pd.DataFrame(columns=["hadm_id"])
print(f"  Imaging features: {img_feat.shape}")

con.close()


# ── 3. Merge all feature sets ──────────────────────────────────────────────────
print("\nMerging feature sets...")

# Base: only ED-linked admissions
base = (cohort[["hadm_id", "subject_id"]]
        .merge(labels, on=["hadm_id", "subject_id"])
        .merge(triage_df[["hadm_id"]], on="hadm_id")  # restrict to ED cohort
       )
print(f"  ED cohort base: {len(base):,} admissions")

# All hadm_ids in ED cohort
all_hadm = base[["hadm_id"]].copy()

# Fill missing: add zero rows for admissions without imaging/lab data
lab_ord_feat  = all_hadm.merge(lab_ord_feat,  on="hadm_id", how="left").fillna(0)
img_feat      = all_hadm.merge(img_feat,       on="hadm_id", how="left").fillna(0)
triage_merged = all_hadm.merge(triage_df,      on="hadm_id", how="left")

def build_ed(use_triage=True, use_lab_ord=True, use_imaging=True):
    df = base.merge(static, on="hadm_id", how="left")
    if use_triage:
        df = df.merge(triage_merged, on="hadm_id", how="left")
    if use_lab_ord:
        df = df.merge(lab_ord_feat,  on="hadm_id", how="left")
    if use_imaging:
        df = df.merge(img_feat,      on="hadm_id", how="left")
    return df


df_vitals  = build_ed(use_triage=True,  use_lab_ord=False, use_imaging=False)
df_lab_ord = build_ed(use_triage=False, use_lab_ord=True,  use_imaging=False)
df_full    = build_ed(use_triage=True,  use_lab_ord=True,  use_imaging=True)

datasets = {
    "triage_vitals_only": df_vitals,
    "ed_lab_ordering":    df_lab_ord,
    "ed_full_behavioral": df_full,
}

for name, df in datasets.items():
    n_feat = len([c for c in df.columns if c not in META])
    print(f"  {name:<24}: {n_feat} features, {len(df):,} rows")


# ── 4. Train/val/test split (patient-level, same seed) ───────────────────────
def patient_split(df):
    pat = df.groupby("subject_id")["mortality"].max().reset_index()
    pat = pat.sample(frac=1, random_state=SEED)
    n    = len(pat)
    n_tr = int(0.70 * n); n_va = int(0.15 * n)
    tr_s = set(pat.iloc[:n_tr]["subject_id"])
    va_s = set(pat.iloc[n_tr:n_tr+n_va]["subject_id"])
    te_s = set(pat.iloc[n_tr+n_va:]["subject_id"])
    return (df["subject_id"].isin(tr_s),
            df["subject_id"].isin(va_s),
            df["subject_id"].isin(te_s))

tr, va, te = patient_split(df_full)
print(f"\nSplit — Train {tr.sum():,} | Val {va.sum():,} | Test {te.sum():,}")
for task in TASKS:
    print(f"  {task}: prev_test={df_full.loc[te, task].mean():.3f}")


# ── 5. LightGBM training ───────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.05, num_leaves=127,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=SEED, verbose=-1, metric="auc",
)

rng = np.random.default_rng(SEED)


def bootstrap_auroc(y, probs, n_boot=N_BOOT):
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() < 2:
            continue
        scores.append(roc_auc_score(y[idx], probs[idx]))
    return (float(np.mean(scores)),
            float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


def train_eval(df, task, tr_mask, va_mask, te_mask, name):
    feat = [c for c in df.columns if c not in META]
    X_tr = df.loc[tr_mask, feat].astype("float32")
    y_tr = df.loc[tr_mask, task]
    X_va = df.loc[va_mask, feat].astype("float32")
    y_va = df.loc[va_mask, task]
    X_te = df.loc[te_mask, feat].astype("float32")
    y_te = df.loc[te_mask, task]
    ok_tr = y_tr.notna(); ok_va = y_va.notna(); ok_te = y_te.notna()
    pos_w = (1 - y_tr[ok_tr].mean()) / y_tr[ok_tr].mean()
    m = lgb.LGBMClassifier(scale_pos_weight=pos_w, **LGBM_PARAMS)
    m.fit(X_tr[ok_tr], y_tr[ok_tr],
          eval_set=[(X_va[ok_va], y_va[ok_va])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    probs = m.predict_proba(X_te[ok_te])[:, 1]
    y_arr = y_te[ok_te].values
    auc, lo, hi = bootstrap_auroc(y_arr, probs)
    print(f"    {name:<24}: {auc:.4f} [{lo:.4f}–{hi:.4f}]")
    return {"auroc": round(auc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


print("\nTraining ED triage models...")
all_results = {}

for task in TASKS:
    print(f"\n  === {task} ===")
    task_res = {}
    for name, df in datasets.items():
        # Re-derive split masks per dataset (same subject_ids, consistent rows)
        tr_t = df["subject_id"].isin(
            df_full.loc[tr, "subject_id"].values)
        va_t = df["subject_id"].isin(
            df_full.loc[va, "subject_id"].values)
        te_t = df["subject_id"].isin(
            df_full.loc[te, "subject_id"].values)
        task_res[name] = train_eval(df, task, tr_t, va_t, te_t, name)

    all_results[task] = task_res
    all_results[task]["N_test"] = int(te.sum())
    all_results[task]["prevalence_test"] = round(
        float(df_full.loc[te, task].dropna().mean()), 4)


# ── 6. Save results ────────────────────────────────────────────────────────────
with open(OUT / "ed_triage_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

rows_csv = []
for task, r in all_results.items():
    for model in list(datasets.keys()):
        if model not in r:
            continue
        rows_csv.append({
            "task": task, "model": model,
            "auroc": r[model]["auroc"],
            "ci_lo": r[model]["ci_lo"],
            "ci_hi": r[model]["ci_hi"],
            "N_test": r.get("N_test", ""),
        })
pd.DataFrame(rows_csv).to_csv(OUT / "ed_triage_auroc.csv", index=False)
print(f"\nSaved ed_triage_results.json + ed_triage_auroc.csv")


# ── 7. Summary table ───────────────────────────────────────────────────────────
# Load inpatient ordering reference AUROCs
RAND_CI = ROOT / "1_ordering_paper" / "results" / "F_paper_improvements" / "bootstrap_ci_results.json"
L_TRIAGE = ROOT / "1_ordering_paper" / "results" / "L_triage_window" / "L_triage_summary.json"

try:
    with open(RAND_CI) as f:  rand_ci = json.load(f)
except: rand_ci = {}
try:
    with open(L_TRIAGE) as f: l_triage = json.load(f)
except: l_triage = {}

print("\n" + "="*75)
print("ED TRIAGE VALIDATION — COMPARISON TO INPATIENT BASELINES")
print("="*75)
for task in TASKS:
    r = all_results[task]
    inp_ord  = rand_ci.get(task, {}).get("ordering_only", {}).get("auroc", "—")
    l_t0     = l_triage.get("triage_auroc", {}).get(task, {}).get("ordering_only", {}).get("auroc", "—")
    print(f"\n  {task}  (test N={r.get('N_test',''):,}, prev={r.get('prevalence_test',0):.1%})")
    for name in datasets:
        auc = r[name]["auroc"]; lo = r[name]["ci_lo"]; hi = r[name]["ci_hi"]
        print(f"    {name:<24}: {auc:.4f} [{lo:.4f}–{hi:.4f}]")
    if isinstance(inp_ord, float):
        print(f"    {'inpatient ordering_only':<24}: {inp_ord:.4f}  [reference]")
    if isinstance(l_t0, float):
        print(f"    {'inpatient triage t=0-4h':<24}: {l_t0:.4f}  [L experiment reference]")
print("="*75)


# ── 8. Figure ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
SKY    = "#56B4E9"

TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "aki":             "AKI",
    "sepsis":          "Sepsis",
}

MODEL_DISPLAY = {
    "triage_vitals_only": ("Triage vitals\n(HR/RR/SpO₂/SBP…)", SKY,    None),
    "ed_lab_ordering":    ("ED lab ordering\n(no values)",       BLUE,   "///"),
    "ed_full_behavioral": ("ED full behavioral\n(labs+img+meds)", GREEN, "///"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

print("\nBuilding Figure 15 — ED Triage Validation...")

fig, axes = plt.subplots(1, 4, figsize=(16, 6))
fig.suptitle(
    "Figure 15 — Emergency Department Triage Validation\n"
    "Predicting inpatient outcomes from ED ordering behavior "
    "before any results are available",
    fontsize=11, fontweight="bold", y=1.03
)

x = np.arange(len(MODEL_DISPLAY))
w = 0.55

for ax_idx, task in enumerate(TASKS):
    ax = axes[ax_idx]
    r  = all_results[task]

    aurocs  = [r[m]["auroc"] for m in MODEL_DISPLAY]
    lo_errs = [r[m]["auroc"] - r[m]["ci_lo"] for m in MODEL_DISPLAY]
    hi_errs = [r[m]["ci_hi"] - r[m]["auroc"] for m in MODEL_DISPLAY]
    colors  = [c for _, c, _ in MODEL_DISPLAY.values()]
    hatches = [h for _, _, h in MODEL_DISPLAY.values()]

    bars = ax.bar(x, aurocs, width=w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for b, hatch in zip(bars, hatches):
        if hatch:
            b.set_hatch(hatch)
    ax.errorbar(x, aurocs, yerr=[lo_errs, hi_errs],
                fmt="none", color="#333333", capsize=4, linewidth=1.3)

    for xi, (a, he) in enumerate(zip(aurocs, hi_errs)):
        ax.text(xi, a + he + 0.004, f"{a:.3f}",
                ha="center", fontsize=8, fontweight="bold", color="#333333")

    # Reference lines: inpatient ordering and L triage
    inp_ref = rand_ci.get(task, {}).get("ordering_only", {}).get("auroc")
    l_ref   = (l_triage.get("triage_auroc", {})
                       .get(task, {})
                       .get("ordering_only", {})
                       .get("auroc"))
    if isinstance(inp_ref, float):
        ax.axhline(inp_ref, color=BLUE, linewidth=1.5, linestyle="--",
                   alpha=0.6, label=f"Inpatient ord. {inp_ref:.3f}")
        ax.text(len(MODEL_DISPLAY) - 0.3, inp_ref + 0.003,
                f"Inpatient\n{inp_ref:.3f}", fontsize=6.5, color=BLUE,
                ha="right", va="bottom")
    if isinstance(l_ref, float):
        ax.axhline(l_ref, color=ORANGE, linewidth=1.5, linestyle=":",
                   alpha=0.7, label=f"Inpatient triage {l_ref:.3f}")
        ax.text(len(MODEL_DISPLAY) - 0.3, l_ref - 0.012,
                f"IP triage\n{l_ref:.3f}", fontsize=6.5, color=ORANGE,
                ha="right", va="top")

    ymin = max(0.50, min(aurocs) - 0.06)
    ymax = max(aurocs) + max(hi_errs) + 0.06
    if isinstance(inp_ref, float):
        ymax = max(ymax, inp_ref + 0.04)
    ax.set_ylim(ymin, min(ymax, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in MODEL_DISPLAY.values()],
                       fontsize=8, ha="center")
    ax.set_ylabel("AUROC" if ax_idx == 0 else "")
    ax.set_title(TASK_LABELS[task])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.text(0.98, 0.03,
            f"N={r.get('N_test',0):,}\nPrev={r.get('prevalence_test',0):.1%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")

patches = [mpatches.Patch(color=c, alpha=0.85, label=l)
           for _, (l, c, _) in MODEL_DISPLAY.items()]
patches += [
    Line2D([0], [0], color=BLUE,   linewidth=1.5, linestyle="--",
           alpha=0.6, label="Inpatient ordering-only (reference)"),
    Line2D([0], [0], color=ORANGE, linewidth=1.5, linestyle=":",
           alpha=0.7, label="Inpatient triage t=0–4h (reference)"),
]
fig.legend(handles=patches, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig(FIG / "fig15_ed_triage.png", bbox_inches="tight")
plt.close()
print("  → fig15_ed_triage.png")

print(f"\nAll ED triage outputs → {OUT}")
