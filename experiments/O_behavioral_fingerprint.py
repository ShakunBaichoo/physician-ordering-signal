#!/usr/bin/env python
"""
O_behavioral_fingerprint.py
============================
Extends the "ordering signal" from lab tests to the full physician
behavioral fingerprint in the EHR:

  (1) Lab ordering — existing signal (from poe, order_type='Lab')
  (2) Imaging ordering — XRay, CT, MRI, Ultrasound, Echo, ECG
  (3) Medication class ordering — antibiotics, vasopressors, diuretics,
      anticoagulants, opioids, sedatives, steroids, IV fluids, electrolytes

Research question:
  Does the broader physician behavioral signal — including what studies
  and drugs are ordered, not just lab tests — provide additional predictive
  power beyond lab ordering alone?

Models compared:
  A) lab_ordering_only   — existing signal (replicates Exp A)
  B) imaging_ordering    — radiology + cardiology orders only
  C) med_class_ordering  — medication class counts only
  D) full_behavioral     — lab + imaging + medication (all behavioral)
  E) values_only         — lab/vital values (existing baseline)
  F) behavioral_plus_values — full behavioral + values

Key expected finding:
  Imaging and medication ordering provide INDEPENDENT signal beyond lab
  ordering, upgrading the claim from "lab ordering" to "physician
  behavioral encoding broadly."

Outputs → 1_ordering_paper/results/O_behavioral/
  behavioral_results.json
  behavioral_auroc.csv
  shapley_behavioral.csv    (cooperative Shapley across 3 modalities)
  fig14_behavioral.png
"""

import json
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
DB   = ROOT / "data" / "raw" / "mimic_iv_2_2.db"
OUT  = ROOT / "1_ordering_paper" / "results" / "O_behavioral"
FIG  = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED   = 42
N_BOOT = 1000
N_BINS = 12       # 4-hour bins over 48h
BIN_HRS = 4       # hours per bin

# ── Drug class definitions (keyword matching on drug name) ────────────────────
DRUG_CLASSES = {
    "rx_antibiotic": [
        "vancomycin", "piperacillin", "meropenem", "cefepime", "ceftriaxone",
        "metronidazole", "ciprofloxacin", "azithromycin", "linezolid",
        "daptomycin", "ampicillin", "nafcillin", "clindamycin", "trimethoprim",
        "levofloxacin", "fluconazole", "micafungin", "caspofungin", "acyclovir",
        "gentamicin", "tobramycin", "imipenem", "ertapenem", "tigecycline",
    ],
    "rx_vasopressor": [
        "norepinephrine", "vasopressin", "phenylephrine", "dopamine",
        "epinephrine", "milrinone", "dobutamine", "neosynephrine",
    ],
    "rx_diuretic": [
        "furosemide", "torsemide", "bumetanide", "metolazone",
        "hydrochlorothiazide", "spironolactone", "acetazolamide",
    ],
    "rx_anticoagulant": [
        "heparin", "warfarin", "enoxaparin", "fondaparinux",
        "argatroban", "bivalirudin", "rivaroxaban", "apixaban",
    ],
    "rx_opioid": [
        "morphine", "fentanyl", "hydromorphone", "dilaudid", "oxycodone",
        "oxycontin", "hydrocodone", "methadone", "meperidine",
    ],
    "rx_sedative": [
        "propofol", "midazolam", "lorazepam", "dexmedetomidine", "precedex",
        "ketamine", "etomidate", "haloperidol", "quetiapine",
    ],
    "rx_steroid": [
        "methylprednisolone", "dexamethasone", "hydrocortisone",
        "prednisone", "prednisolone", "solumedrol",
    ],
    "rx_insulin": ["insulin"],
    "rx_ivfluid": [
        "0.9% sodium chloride", "normal saline", "lactated ringers",
        "5% dextrose", "albumin", "plasma-lyte",
    ],
    "rx_electrolyte": [
        "potassium chloride", "magnesium sulfate", "calcium gluconate",
        "calcium chloride", "sodium bicarbonate", "phosphate",
    ],
}

# Imaging category mapping from poe subtype
IMAGING_CATS = {
    "xray":         ["General Xray"],
    "ct_scan":      ["CT Scan"],
    "mri":          ["MRI"],
    "ultrasound":   ["Ultrasound", "Noninvasive Vascular"],
    "ir":           ["Interventional Radiology", "Cross-Sectional Interventional Radiology",
                     "Interventional Neuro", "Angio"],
    "nuclear_med":  ["Nuclear Med"],
    "ecg":          ["ECG"],   # from Cardiology
    "echo":         ["Echo", "Stress Echo"],  # from Cardiology
}


# ── 1. Load cohort & labels ────────────────────────────────────────────────────
print("Loading cohort, labels, static...")
cohort = pd.read_parquet(DATA / "cohort.parquet")[
    ["hadm_id", "subject_id", "admittime", "dischtime"]
]
cohort["admittime_dt"] = pd.to_datetime(cohort["admittime"])
cohort_hids = set(cohort["hadm_id"])

labels = pd.read_parquet(DATA / "labels.parquet")
static = pd.read_parquet(DATA / "static.parquet")

META = ["hadm_id", "subject_id"] + TASKS
print(f"  Cohort: {len(cohort):,} admissions")


# ── 2. Extract non-lab behavioral features from POE ───────────────────────────
print("\nQuerying POE for imaging + cardiology orders...")
con = duckdb.connect()
con.execute(f"ATTACH '{DB}' AS mimic (TYPE SQLITE, READ_ONLY TRUE)")

# Build mapping subtype → imaging_cat
sub2cat = {}
for cat, subtypes in IMAGING_CATS.items():
    for st in subtypes:
        sub2cat[st] = cat

poe_q = """
    SELECT p.hadm_id,
           p.ordertime,
           p.order_type,
           p.order_subtype
    FROM mimic.poe p
    WHERE p.order_type IN ('Radiology', 'Cardiology')
      AND p.hadm_id IS NOT NULL
"""
poe_df = con.execute(poe_q).df()
print(f"  Imaging/Cardiology POE rows: {len(poe_df):,}")

# Medications: pull prescriptions with starttime
print("Querying prescriptions...")
rx_q = """
    SELECT hadm_id, starttime, LOWER(drug) AS drug_lower
    FROM mimic.prescriptions
    WHERE hadm_id IS NOT NULL
"""
rx_df = con.execute(rx_q).df()
print(f"  Prescriptions rows: {len(rx_df):,}")

con.close()


# ── 3. Build imaging features (count per category per 4h bin) ─────────────────
print("\nBuilding imaging features...")

# Filter to cohort, parse times
poe_df = poe_df[poe_df["hadm_id"].isin(cohort_hids)].copy()
poe_df["ordertime_dt"] = pd.to_datetime(poe_df["ordertime"], errors="coerce")

# Merge admittime
poe_df = poe_df.merge(cohort[["hadm_id", "admittime_dt"]], on="hadm_id", how="left")
poe_df["hours_from_admit"] = (
    (poe_df["ordertime_dt"] - poe_df["admittime_dt"]).dt.total_seconds() / 3600
)
poe_df = poe_df[(poe_df["hours_from_admit"] >= 0) & (poe_df["hours_from_admit"] < 48)]
poe_df["time_bin"] = (poe_df["hours_from_admit"] / BIN_HRS).astype(int).clip(0, N_BINS - 1)

# Map to imaging category
poe_df["img_cat"] = poe_df["order_subtype"].map(sub2cat)
poe_df = poe_df.dropna(subset=["img_cat"])

print(f"  Imaging rows in first 48h: {len(poe_df):,}")
print(f"  Imaging category counts:\n{poe_df['img_cat'].value_counts().to_string()}")

# Pivot: count per hadm_id × img_cat × time_bin
def build_img_features(df, categories):
    """Create wide feature matrix: img_cat_total + img_cat_intensity + per-bin counts."""
    rows = []
    for hadm_id, grp in df.groupby("hadm_id"):
        row = {"hadm_id": hadm_id}
        for cat in categories:
            sub = grp[grp["img_cat"] == cat]
            total = len(sub)
            row[f"{cat}_total"] = total
            row[f"{cat}_intensity"] = total / N_BINS
            for b in range(N_BINS):
                row[f"{cat}_bin{b}"] = int((sub["time_bin"] == b).sum())
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    feat_df = pd.DataFrame(rows)
    # Admission-level aggregates
    all_cats = list(categories)
    feat_df["img_total_orders"] = feat_df[[f"{c}_total" for c in all_cats]].sum(axis=1)
    feat_df["img_diversity"] = (feat_df[[f"{c}_total" for c in all_cats]] > 0).sum(axis=1)
    return feat_df


IMG_CATS = list(IMAGING_CATS.keys())
img_feat = build_img_features(poe_df, IMG_CATS)
print(f"  Imaging feature matrix: {img_feat.shape} (admissions with any imaging: {(img_feat['img_total_orders'] > 0).sum():,})")

# Add zero rows for admissions with NO imaging orders
all_hadm = cohort[["hadm_id"]].copy()
img_feat = all_hadm.merge(img_feat, on="hadm_id", how="left").fillna(0)
print(f"  Imaging feature matrix (all admissions): {img_feat.shape}")


# ── 4. Build medication class features ────────────────────────────────────────
print("\nBuilding medication class features...")

rx_df = rx_df[rx_df["hadm_id"].isin(cohort_hids)].copy()
rx_df["starttime_dt"] = pd.to_datetime(rx_df["starttime"], errors="coerce")
rx_df = rx_df.merge(cohort[["hadm_id", "admittime_dt"]], on="hadm_id", how="left")
rx_df["hours_from_admit"] = (
    (rx_df["starttime_dt"] - rx_df["admittime_dt"]).dt.total_seconds() / 3600
)
rx_df = rx_df[(rx_df["hours_from_admit"] >= 0) & (rx_df["hours_from_admit"] < 48)]
rx_df["time_bin"] = (rx_df["hours_from_admit"] / BIN_HRS).astype(int).clip(0, N_BINS - 1)

# Classify each prescription into drug class
def classify_drug(drug_lower, classes):
    if not drug_lower or not isinstance(drug_lower, str):
        return None
    for cls, keywords in classes.items():
        if any(kw in drug_lower for kw in keywords):
            return cls
    return None

print("  Classifying drugs (vectorised)...")
rx_df["drug_class"] = rx_df["drug_lower"].apply(
    lambda d: classify_drug(d, DRUG_CLASSES)
)
classified = rx_df.dropna(subset=["drug_class"])
print(f"  Classified {len(classified):,}/{len(rx_df):,} prescriptions")
print(f"  Drug class distribution:\n{classified['drug_class'].value_counts().to_string()}")


def build_rx_features(df, classes):
    """Count per hadm_id × drug_class × time_bin."""
    rows = []
    for hadm_id, grp in df.groupby("hadm_id"):
        row = {"hadm_id": hadm_id}
        for cls in classes:
            sub = grp[grp["drug_class"] == cls]
            total = len(sub)
            row[f"{cls}_total"] = total
            row[f"{cls}_intensity"] = total / N_BINS
            for b in range(N_BINS):
                row[f"{cls}_bin{b}"] = int((sub["time_bin"] == b).sum())
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    feat_df = pd.DataFrame(rows)
    cls_list = list(classes.keys())
    feat_df["rx_total_orders"] = feat_df[[f"{c}_total" for c in cls_list]].sum(axis=1)
    feat_df["rx_n_classes"]    = (feat_df[[f"{c}_total" for c in cls_list]] > 0).sum(axis=1)
    feat_df["rx_has_vasopressor"] = (feat_df["rx_vasopressor_total"] > 0).astype(int)
    feat_df["rx_has_antibiotic"]  = (feat_df["rx_antibiotic_total"] > 0).astype(int)
    feat_df["rx_has_anticoag"]    = (feat_df["rx_anticoagulant_total"] > 0).astype(int)
    return feat_df


rx_feat = build_rx_features(classified, DRUG_CLASSES)
rx_feat = all_hadm.merge(rx_feat, on="hadm_id", how="left").fillna(0)
print(f"  Medication feature matrix: {rx_feat.shape}")


# ── 5. Load existing lab ordering + value features ────────────────────────────
print("\nLoading lab ordering + value features...")
labs_long   = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")


def value_features(df_long, name):
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id", "time_bin") and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()


def ordering_features(df_long, name):
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    if not obs_cols:
        return pd.DataFrame({"hadm_id": df_long["hadm_id"].unique()})

    print(f"  Ordering features for {name}...")
    g = df_long.groupby("hadm_id", sort=False)
    total     = g[obs_cols].sum();     total.columns     = [f"{c}_total"     for c in obs_cols]
    intensity = g[obs_cols].mean();    intensity.columns = [f"{c}_intensity" for c in obs_cols]
    obs_data  = df_long[["hadm_id"] + obs_cols].copy()
    obs_data["any_obs"]    = (obs_data[obs_cols] > 0).any(axis=1).astype(int)
    obs_data["n_tests_bin"] = (obs_data[obs_cols] > 0).sum(axis=1)
    adm_int   = obs_data.groupby("hadm_id")["any_obs"].mean().rename("adm_ordering_intensity")
    diversity = obs_data.groupby("hadm_id")["n_tests_bin"].mean().rename("adm_ordering_diversity")
    breadth   = (g[obs_cols].sum() > 0).sum(axis=1).rename("adm_ordering_breadth")
    mid       = df_long.groupby("hadm_id")["time_bin"].transform("median")
    df_long["is_late"] = (df_long["time_bin"] >= mid).astype(int)
    early_int = df_long[df_long["is_late"] == 0].groupby("hadm_id")[obs_cols].mean().mean(axis=1).rename("early_intensity")
    late_int  = df_long[df_long["is_late"] == 1].groupby("hadm_id")[obs_cols].mean().mean(axis=1).rename("late_intensity")
    escalation = (late_int / (early_int + 1e-8)).rename("adm_ordering_escalation")
    df_long.drop(columns=["is_late"], inplace=True)
    agg = total.join(intensity).join(adm_int).join(diversity).join(breadth).join(escalation)
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()


labs_val   = value_features(labs_long,   "lab")
vitals_val = value_features(vitals_long, "vit")
labs_ord   = ordering_features(labs_long,   "lab")
vitals_ord = ordering_features(vitals_long, "vit")


# ── 6. Build feature matrices ─────────────────────────────────────────────────
print("\nBuilding merged feature matrices...")
base = cohort[["hadm_id", "subject_id"]].merge(labels, on=["hadm_id", "subject_id"])
META_COLS = ["hadm_id", "subject_id"] + TASKS


def build(lab_ord=True, lab_val=True, imaging=True, rx=True):
    df = base.merge(static, on="hadm_id", how="left")
    if lab_ord:
        df = df.merge(labs_ord,   on="hadm_id", how="left")
        df = df.merge(vitals_ord, on="hadm_id", how="left")
    if lab_val:
        df = df.merge(labs_val,   on="hadm_id", how="left")
        df = df.merge(vitals_val, on="hadm_id", how="left")
    if imaging:
        df = df.merge(img_feat,   on="hadm_id", how="left")
    if rx:
        df = df.merge(rx_feat,    on="hadm_id", how="left")
    return df


df_lab_ord  = build(lab_ord=True,  lab_val=False, imaging=False, rx=False)
df_img      = build(lab_ord=False, lab_val=False, imaging=True,  rx=False)
df_rx       = build(lab_ord=False, lab_val=False, imaging=False, rx=True)
df_full_beh = build(lab_ord=True,  lab_val=False, imaging=True,  rx=True)
df_val_only = build(lab_ord=False, lab_val=True,  imaging=False, rx=False)
df_beh_val  = build(lab_ord=True,  lab_val=True,  imaging=True,  rx=True)

datasets = {
    "lab_ordering_only":    df_lab_ord,
    "imaging_ordering":     df_img,
    "med_class_ordering":   df_rx,
    "full_behavioral":      df_full_beh,
    "values_only":          df_val_only,
    "behavioral_plus_values": df_beh_val,
}
for name, df in datasets.items():
    n_feat = len([c for c in df.columns if c not in META_COLS])
    print(f"  {name:<28}: {n_feat} features")


# ── 7. Train/val/test split (same patient-level random split as Exp A/H) ──────
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

tr, va, te = patient_split(df_beh_val)
print(f"\nSplit — Train {tr.sum():,} | Val {va.sum():,} | Test {te.sum():,}")


# ── 8. LightGBM training ───────────────────────────────────────────────────────
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


def train_eval(df_dict, task, tr_mask, va_mask, te_mask):
    out = {}
    for name, df in df_dict.items():
        feat = [c for c in df.columns if c not in META_COLS]
        X_tr, y_tr = df.loc[tr_mask, feat].astype("float32"), df.loc[tr_mask, task]
        X_va, y_va = df.loc[va_mask, feat].astype("float32"), df.loc[va_mask, task]
        X_te, y_te = df.loc[te_mask, feat].astype("float32"), df.loc[te_mask, task]
        ok_tr = y_tr.notna(); ok_va = y_va.notna(); ok_te = y_te.notna()
        pos_w = (1 - y_tr[ok_tr].mean()) / y_tr[ok_tr].mean()
        m = lgb.LGBMClassifier(scale_pos_weight=pos_w, **LGBM_PARAMS)
        m.fit(X_tr[ok_tr], y_tr[ok_tr],
              eval_set=[(X_va[ok_va], y_va[ok_va])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        probs = m.predict_proba(X_te[ok_te])[:, 1]
        y_arr = y_te[ok_te].values
        auc, lo, hi = bootstrap_auroc(y_arr, probs)
        out[name] = {"auroc": round(auc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
        print(f"    {name:<28}: {auc:.4f} [{lo:.4f}–{hi:.4f}]")
    return out


# For the split, all datasets must use the same tr/va/te masks
# (they all have same subject_id column from same base)
# We need to re-derive masks per dataset since row counts may differ
# Actually all datasets share the same base rows — use df_beh_val masks

print("\nTraining all behavioral models...")
all_results = {}

for task in TASKS:
    print(f"\n  === {task} ===")
    all_results[task] = train_eval(datasets, task, tr, va, te)
    all_results[task]["N_test"] = int(te.sum())
    all_results[task]["prevalence_test"] = round(
        float(df_beh_val.loc[te, task].dropna().mean()), 4
    )


# ── 9. Save results ────────────────────────────────────────────────────────────
with open(OUT / "behavioral_results.json", "w") as f:
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
            "N_test": r["N_test"],
        })
pd.DataFrame(rows_csv).to_csv(OUT / "behavioral_auroc.csv", index=False)
print(f"\nSaved behavioral_results.json + behavioral_auroc.csv")


# ── 10. Print summary ──────────────────────────────────────────────────────────
print("\n" + "="*80)
print("BEHAVIORAL FINGERPRINT — AUROC SUMMARY")
print("="*80)
model_names = list(datasets.keys())
header = f"{'Model':<28}" + "".join(f"  {t[:8]:>10}" for t in TASKS)
print(header)
print("-"*80)
for model in model_names:
    row = f"{model:<28}"
    for task in TASKS:
        if model in all_results[task]:
            row += f"  {all_results[task][model]['auroc']:>10.4f}"
        else:
            row += f"  {'—':>10}"
    print(row)
print("="*80)

# AUROC increments over lab ordering
print("\nIncremental AUROC over lab_ordering_only:")
for model in model_names:
    if model == "lab_ordering_only":
        continue
    row = f"  {model:<28}"
    for task in TASKS:
        if model in all_results[task] and "lab_ordering_only" in all_results[task]:
            delta = all_results[task][model]["auroc"] - all_results[task]["lab_ordering_only"]["auroc"]
            row += f"  {delta:>+10.4f}"
    print(row)
print("="*80)


# ── 11. Figure ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
SKY    = "#56B4E9"
PURPLE = "#CC79A7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "aki":             "AKI",
    "sepsis":          "Sepsis",
}

MODEL_DISPLAY = {
    "lab_ordering_only":      ("Lab ordering only",    BLUE,   "///"),
    "imaging_ordering":       ("Imaging ordering",     SKY,    None),
    "med_class_ordering":     ("Medication ordering",  PURPLE, None),
    "full_behavioral":        ("Full behavioral",      GREEN,  "///"),
    "values_only":            ("Values only",          ORANGE, None),
    "behavioral_plus_values": ("Behavioral + values",  RED,    None),
}

print("\nBuilding Figure 14 — Behavioral Fingerprint...")

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle(
    "Figure 14 — Full Physician Behavioral Fingerprint: Lab, Imaging, and Medication Ordering\n"
    "Extending the ordering signal beyond laboratory tests to all EHR order categories",
    fontsize=11, fontweight="bold", y=1.03
)

x = np.arange(len(MODEL_DISPLAY))
w = 0.65

for ax_idx, task in enumerate(TASKS):
    ax = axes[ax_idx]
    aurocs = []; lo_errs = []; hi_errs = []; colors = []; hatches = []

    for model, (label, color, hatch) in MODEL_DISPLAY.items():
        r = all_results[task].get(model, {})
        aurocs.append(r.get("auroc", np.nan))
        lo_errs.append(r.get("auroc", 0) - r.get("ci_lo", r.get("auroc", 0)))
        hi_errs.append(r.get("ci_hi", r.get("auroc", 0)) - r.get("auroc", 0))
        colors.append(color)
        hatches.append(hatch)

    bars = ax.bar(x, aurocs, width=w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for b, hatch in zip(bars, hatches):
        if hatch:
            b.set_hatch(hatch)
    ax.errorbar(x, aurocs, yerr=[lo_errs, hi_errs],
                fmt="none", color="#333333", capsize=3, linewidth=1.2)

    for xi, (a, he) in enumerate(zip(aurocs, hi_errs)):
        if not np.isnan(a):
            ax.text(xi, a + he + 0.003, f"{a:.3f}",
                    ha="center", fontsize=6.5, fontweight="bold", color="#333333")

    ymin = max(0.50, min(a for a in aurocs if not np.isnan(a)) - 0.05)
    ymax = max(a for a in aurocs if not np.isnan(a)) + max(hi_errs) + 0.04
    ax.set_ylim(ymin, min(ymax + 0.01, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in MODEL_DISPLAY.values()],
                       rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("AUROC" if ax_idx == 0 else "")
    ax.set_title(TASK_LABELS[task])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

patches = [mpatches.Patch(color=c, alpha=0.85, label=l)
           for _, (l, c, _) in MODEL_DISPLAY.items()]
fig.legend(handles=patches, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(FIG / "fig14_behavioral.png", bbox_inches="tight")
plt.close()
print("  → fig14_behavioral.png")

print(f"\nAll behavioral fingerprint outputs → {OUT}")
