#!/usr/bin/env python
"""
Experiment Q — MC-MED External Validation
==========================================
Validate the ED ordering signal on MC-MED (Stanford, 118K ED visits, 2020–2022).
Zero-shot feature engineering: same approach as P_ed_triage_validation.py but
applied to MC-MED orders.csv + numerics.csv + visits.csv.

Outcomes validated:
  - In-hospital mortality  (DC_dispo == "Expired")
  - 30-day readmission     (Hours_to_next_visit < 720)
  - ICU admission          (ED_dispo == "ICU")
  - Inpatient admission    (ED_dispo == "Inpatient")  [bonus]

Models:
  1. triage_vitals_only    — HR/RR/SpO2/SBP/DBP/Temp/acuity/age/sex
  2. ed_ordering_only      — order counts/flags from orders.csv (no values)
  3. ed_full               — vitals + ordering combined
  4. values_only           — lab result values from labs.csv (if available)
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT   = Path(__file__).parents[2]
MCMED  = ROOT / "physionet.org" / "files" / "mc-med" / "1.0.1" / "data"
OUT    = ROOT / "1_ordering_paper" / "results" / "Q_mcmed"
FIG    = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reference results from MIMIC P experiment
MIMIC_P_REF = ROOT / "1_ordering_paper" / "results" / "P_ed_triage" / "ed_triage_results.json"

TASKS = ["mortality", "readmission_30d", "icu_admission", "inpatient_admit"]
TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "icu_admission":   "ICU Admission",
    "inpatient_admit": "Inpatient Admission",
}

# Wong (2011) colorblind-safe palette
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
SKY    = "#56B4E9"

RANDOM_STATE = 42
N_BOOT = 1000


def bootstrap_auroc(y_true, y_prob, n=N_BOOT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    aucs = []
    n_pos = y_true.sum()
    if n_pos < 5 or (len(y_true) - n_pos) < 5:
        return np.nan, np.nan, np.nan
    auc = roc_auc_score(y_true, y_prob)
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, yp))
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_auc_diff(y_true, prob_a, prob_b, n=N_BOOT, seed=RANDOM_STATE):
    """Paired bootstrap test for AUROC(model A) - AUROC(model B)."""
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)
    observed = roc_auc_score(y_true, prob_a) - roc_auc_score(y_true, prob_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        diffs.append(roc_auc_score(yt, prob_a[idx]) - roc_auc_score(yt, prob_b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return float(observed), float(lo), float(hi), float(min(p, 1.0))


def train_lgbm(X_train, y_train, X_val, y_val):
    pos = y_train.sum()
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)
    model = lgb.LGBMClassifier(
        n_estimators=2000, num_leaves=63, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        scale_pos_weight=spw, random_state=RANDOM_STATE,
        metric="auc", n_jobs=8, verbose=-1
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    )
    return model


# ── 1. Load visits ─────────────────────────────────────────────────────────
print("Loading visits.csv ...")
visits = pd.read_csv(MCMED / "visits.csv", low_memory=False)
print(f"  {len(visits):,} visits")

# Parse timestamps
for col in ["Arrival_time", "Departure_time", "Admit_time", "Dispo_time"]:
    visits[col] = pd.to_datetime(visits[col], utc=True, errors="coerce")

# ── 2. Define outcomes ─────────────────────────────────────────────────────
visits["mortality"]       = visits["DC_dispo"].str.contains("Expired", case=False, na=False).astype(int)
visits["readmission_30d"] = ((visits["Hours_to_next_visit"] > 0) &
                              (visits["Hours_to_next_visit"] < 720)).astype(int)
visits["icu_admission"]   = (visits["ED_dispo"] == "ICU").astype(int)
visits["inpatient_admit"] = visits["ED_dispo"].isin(["Inpatient", "ICU"]).astype(int)

print("\nOutcome prevalences:")
for t in TASKS:
    n_pos = visits[t].sum()
    print(f"  {t}: {n_pos:,} / {len(visits):,} ({n_pos/len(visits):.1%})")

# ── 3. Patient-level random split (80 / 10 / 10 by MRN) ───────────────────
# Note: MC-MED timestamps are randomly shifted per patient — a temporal split
# on shifted times is not chronologically meaningful. Use random patient split.
rng = np.random.default_rng(RANDOM_STATE)
unique_mrns = visits["MRN"].unique()
rng.shuffle(unique_mrns)
n_mrn = len(unique_mrns)
train_mrns = set(unique_mrns[:int(n_mrn * 0.80)])
val_mrns   = set(unique_mrns[int(n_mrn * 0.80):int(n_mrn * 0.90)])
test_mrns  = set(unique_mrns[int(n_mrn * 0.90):])

visits = visits.reset_index(drop=True)
idx_train = visits.index[visits["MRN"].isin(train_mrns)]
idx_val   = visits.index[visits["MRN"].isin(val_mrns)]
idx_test  = visits.index[visits["MRN"].isin(test_mrns)]
print(f"\nSplit — train: {len(idx_train):,}  val: {len(idx_val):,}  test: {len(idx_test):,}")

# ── 4. Triage vitals features ──────────────────────────────────────────────
TRIAGE_VITALS = ["Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP",
                 "Triage_DBP", "Triage_Temp"]
TRIAGE_CAT    = ["Triage_acuity", "Gender", "Means_of_arrival"]

visits["age_years"] = visits["Age"].clip(18, 90)
visits["is_female"] = (visits["Gender"] == "F").astype(float)
visits["ed_los_hr"] = visits["ED_LOS"].clip(0, 48)

# Encode acuity (ESI 1–5)
visits["acuity_num"] = pd.to_numeric(
    visits["Triage_acuity"].astype(str).str.extract(r"(\d)")[0], errors="coerce"
)
# Means of arrival
visits["is_ems"] = visits["Means_of_arrival"].str.contains("EMS|Ambulance", case=False, na=False).astype(float)

# Fill missing vitals with median
for col in TRIAGE_VITALS:
    med = visits.loc[idx_train, col].median()
    visits[col] = visits[col].fillna(med)

TRIAGE_FEATURES = TRIAGE_VITALS + ["age_years", "is_female", "acuity_num", "is_ems", "ed_los_hr"]

# ── 5. Load & featurise orders ──────────────────────────────────────────────
print("\nLoading orders.csv ...")
orders = pd.read_csv(MCMED / "orders.csv", low_memory=False)
print(f"  {len(orders):,} orders")

orders["Order_time"] = pd.to_datetime(orders["Order_time"], utc=True, errors="coerce")
orders["proc_lower"] = orders["Procedure_name"].str.lower().fillna("")
orders["type_lower"] = orders["Order_type"].str.lower().fillna("")

# Merge ED window from visits
ed_window = visits[["CSN", "Arrival_time", "Departure_time"]].copy()
orders = orders.merge(ed_window, on="CSN", how="inner")
orders["Departure_time"] = pd.to_datetime(orders["Departure_time"], utc=True, errors="coerce")

# Keep orders placed during ED stay
mask_in_ed = (
    orders["Order_time"].notna() &
    orders["Arrival_time"].notna() &
    (orders["Order_time"] >= orders["Arrival_time"])
)
mask_before_dept = (
    orders["Departure_time"].isna() |
    (orders["Order_time"] <= orders["Departure_time"])
)
orders = orders[mask_in_ed & mask_before_dept].copy()
print(f"  {len(orders):,} orders within ED window")

# ── Order type counts (groupby pivot — most reliable approach) ──────────────
print("  Building per-CSN order features ...")

# By Order_type (pivot: one column per type)
type_counts = (orders.groupby(["CSN", "Order_type"]).size()
               .unstack(fill_value=0))
type_counts.columns = [f"n_otype_{c.lower().replace(' ', '_').replace('/', '_')}"
                       for c in type_counts.columns]

# Procedure-level pattern flags
PROC_PATTERNS = {
    "n_xray":        r"\bxr\b|x.?ray",
    "n_ct":          r"\bct\b|computed tomography",
    "n_mri":         r"\bmri\b|magnetic resonance",
    "n_ultrasound":  r"ultrasound|\bus\b",
    "n_echo":        r"echo(?:cardiogram)?",
    "n_ecg":         r"\becg\b|\bekg\b|electrocardiogram",
    "n_culture":     r"culture",
    "n_cbc":         r"\bcbc\b|complete blood",
    "n_bmp":         r"\bbmp\b|\bcmp\b|metabolic panel|basic metabolic",
    "n_troponin":    r"troponin",
    "n_lactate":     r"lactate",
    "n_coag":        r"\bpt\b|\bptt\b|\binr\b|coagulation|protime",
    "n_lipase":      r"lipase",
    "n_urinalysis":  r"urinalysis|\bua\b",
    "n_abg":         r"\babg\b|arterial blood gas",
    "n_blood_cx":    r"blood culture",
    "n_urine_cx":    r"urine culture",
    "n_antibiotic":  r"ceftriaxone|vancomycin|piperacillin|azithromycin|"
                     r"metronidazole|ciprofloxacin|ampicillin",
    "n_iv_fluid":    r"normal saline|lactated ringer|0\.9%.*nacl",
    "n_vasopressor": r"norepinephrine|vasopressin|phenylephrine",
    "n_opioid":      r"morphine|hydromorphone|fentanyl|oxycodone",
    "n_anticoag":    r"heparin|enoxaparin|warfarin|apixaban|rivaroxaban",
}
proc_series = {}
for feat, pat in PROC_PATTERNS.items():
    mask = orders["proc_lower"].str.contains(pat, regex=True, na=False)
    proc_series[feat] = orders[mask].groupby("CSN").size()
proc_counts = pd.DataFrame(proc_series).fillna(0)

# Total orders
total = orders.groupby("CSN").size().rename("n_total_orders")

# Combine all ordering features
order_feats = type_counts.join(proc_counts, how="outer").join(total, how="outer").fillna(0)
order_feats = order_feats.reset_index()  # CSN → column

ORDER_FEAT_COLS = [c for c in order_feats.columns if c != "CSN"]
print(f"  {len(ORDER_FEAT_COLS)} ordering features")

# Merge into visits
visits = visits.merge(order_feats, on="CSN", how="left")
for c in ORDER_FEAT_COLS:
    visits[c] = visits[c].fillna(0)

# ── 6. Load lab values (if available) ─────────────────────────────────────
LABS_PATH = MCMED / "labs.csv"
has_labs = LABS_PATH.exists()

LAB_VALUE_COLS = []
if has_labs:
    print("\nLoading labs.csv for values-only model ...")
    try:
        labs = pd.read_csv(LABS_PATH, low_memory=False)
        labs["Order_time"] = pd.to_datetime(labs["Order_time"], utc=True, errors="coerce")
        labs["Component_value"] = pd.to_numeric(labs["Component_value"], errors="coerce")
        labs = labs.merge(ed_window, on="CSN", how="inner")
        labs["Departure_time"] = pd.to_datetime(labs["Departure_time"], utc=True, errors="coerce")
        labs_in_ed = labs[
            labs["Order_time"].notna() &
            labs["Arrival_time"].notna() &
            (labs["Order_time"] >= labs["Arrival_time"])
        ]
        top_components = (labs_in_ed.groupby("Component_name")["CSN"]
                          .nunique().sort_values(ascending=False).head(40).index.tolist())
        print(f"  Top {len(top_components)} lab components")
        lab_pivot = (labs_in_ed[labs_in_ed["Component_name"].isin(top_components)]
                     .groupby(["CSN", "Component_name"])["Component_value"]
                     .mean().unstack(fill_value=np.nan))
        lab_pivot.columns = [f"lab_{c.replace(' ', '_').replace(',', '').lower()}"
                             for c in lab_pivot.columns]
        if "Component_abnormal" in labs_in_ed.columns:
            abn = labs_in_ed.copy()
            abn["is_abnormal"] = abn["Component_abnormal"].str.contains(
                "Abnormal|High|Low|Critical", case=False, na=False).astype(float)
            lab_pivot["n_abnormal_results"] = abn.groupby("CSN")["is_abnormal"].sum()
        visits = visits.merge(lab_pivot.reset_index(), on="CSN", how="left")
        LAB_VALUE_COLS = list(lab_pivot.columns)
        for c in LAB_VALUE_COLS:
            med = visits.loc[idx_train, c].median() if c in visits.columns else 0
            visits[c] = visits[c].fillna(med)
        print(f"  {len(LAB_VALUE_COLS)} lab value features")
        has_labs = True
    except Exception as e:
        print(f"  labs.csv error ({e}) — skipping values-only model")
        has_labs = False
else:
    print("\nNo labs.csv — skipping values-only model")

# ── 7. Train & evaluate ─────────────────────────────────────────────────────
all_results = {}
all_predictions = []

for task in TASKS:
    print(f"\n{'='*60}")
    print(f"Task: {task}")

    y = visits[task].values
    y_train = y[idx_train]
    y_val   = y[idx_val]
    y_test  = y[idx_test]

    n_pos = y_test.sum()
    print(f"  Test: N={len(y_test):,}, pos={n_pos} ({n_pos/len(y_test):.1%})")

    task_res = {
        "N_test": int(len(y_test)),
        "prevalence_test": float(n_pos / len(y_test))
    }

    model_specs = {
        "triage_vitals_only": TRIAGE_FEATURES,
        "ed_ordering_only":   ORDER_FEAT_COLS,
        "ed_full":            TRIAGE_FEATURES + ORDER_FEAT_COLS,
    }
    if has_labs and LAB_VALUE_COLS:
        model_specs["values_only"] = TRIAGE_FEATURES[:6] + LAB_VALUE_COLS  # vitals + lab values

    for model_name, feat_cols in model_specs.items():
        feat_cols = [c for c in feat_cols if c in visits.columns]
        X_train = visits.loc[idx_train, feat_cols].values.astype(float)
        X_val   = visits.loc[idx_val,   feat_cols].values.astype(float)
        X_test  = visits.loc[idx_test,  feat_cols].values.astype(float)

        # Replace any remaining NaN
        col_means = np.nanmean(X_train, axis=0)
        col_means = np.nan_to_num(col_means, nan=0.0)
        for arr in [X_train, X_val, X_test]:
            nan_mask = np.isnan(arr)
            if nan_mask.any():
                arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

        nz_cols = int(np.sum(X_train.sum(axis=0) != 0))
        nz_rows = int(np.sum(X_train.sum(axis=1) != 0))
        print(f"    [{model_name}] nonzero cols={nz_cols}/{X_train.shape[1]}, "
              f"nonzero rows={nz_rows}/{X_train.shape[0]}")

        if y_train.sum() < 10:
            print(f"  {model_name}: too few positives in training — skipping")
            continue

        model = train_lgbm(X_train, y_train, X_val, y_val)
        y_prob = model.predict_proba(X_test)[:, 1]
        all_predictions.append(pd.DataFrame({
            "task": task,
            "model": model_name,
            "row_index": idx_test,
            "CSN": visits.loc[idx_test, "CSN"].values,
            "MRN": visits.loc[idx_test, "MRN"].values,
            "y_true": y_test,
            "y_prob": y_prob,
        }))

        auc, lo, hi = bootstrap_auroc(y_test, y_prob)
        print(f"  {model_name:<25}: AUROC={auc:.4f} [{lo:.4f}–{hi:.4f}]")
        task_res[model_name] = {"auroc": round(auc, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}

    all_results[task] = task_res

# ── 8. Save results ─────────────────────────────────────────────────────────
with open(OUT / "mcmed_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved → {OUT / 'mcmed_results.json'}")

# CSV summary
rows = []
for task, r in all_results.items():
    for model, v in r.items():
        if isinstance(v, dict):
            rows.append({"task": task, "model": model,
                         "auroc": v["auroc"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"]})
pd.DataFrame(rows).to_csv(OUT / "mcmed_auroc.csv", index=False)

pred_df = pd.concat(all_predictions, ignore_index=True)
pred_df.to_parquet(OUT / "mcmed_test_predictions.parquet", index=False)

comparison_specs = [
    ("ed_ordering_only", "triage_vitals_only"),
    ("ed_ordering_only", "values_only"),
    ("ed_full", "ed_ordering_only"),
    ("ed_full", "triage_vitals_only"),
    ("ed_full", "values_only"),
    ("triage_vitals_only", "values_only"),
]
comparison_rows = []
for task in TASKS:
    wide = pred_df[pred_df["task"] == task].pivot_table(
        index=["row_index", "CSN", "MRN", "y_true"],
        columns="model",
        values="y_prob",
        aggfunc="first",
    ).reset_index()
    for model_a, model_b in comparison_specs:
        if model_a not in wide.columns or model_b not in wide.columns:
            continue
        sub = wide.dropna(subset=[model_a, model_b, "y_true"])
        if len(sub) == 0 or sub["y_true"].sum() < 5:
            continue
        diff, lo, hi, p = paired_bootstrap_auc_diff(
            sub["y_true"].values, sub[model_a].values, sub[model_b].values
        )
        comparison_rows.append({
            "dataset": "MC-MED held-out test",
            "task": task,
            "model_a": model_a,
            "model_b": model_b,
            "comparison": f"{model_a} - {model_b}",
            "n": int(len(sub)),
            "n_pos": int(sub["y_true"].sum()),
            "auc_a": float(roc_auc_score(sub["y_true"], sub[model_a])),
            "auc_b": float(roc_auc_score(sub["y_true"], sub[model_b])),
            "auc_diff": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "p_value_two_sided": p,
        })
pd.DataFrame(comparison_rows).to_csv(OUT / "mcmed_paired_auc_comparisons.csv", index=False)
print(f"Saved → {OUT / 'mcmed_test_predictions.parquet'}")
print(f"Saved → {OUT / 'mcmed_paired_auc_comparisons.csv'}")

# ── 9. Figure ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Load MIMIC P reference
try:
    with open(MIMIC_P_REF) as f:
        mimic_p = json.load(f)
except Exception:
    mimic_p = {}

MODEL_DISPLAY = {
    "triage_vitals_only": ("Triage vitals\n(NEWS2-like)",   SKY,    None),
    "ed_ordering_only":   ("ED ordering\n(no values)",      BLUE,   "///"),
    "ed_full":            ("ED full\n(orders+vitals)",       GREEN,  "///"),
}
if has_labs and LAB_VALUE_COLS:
    MODEL_DISPLAY["values_only"] = ("Lab values\n(results)", ORANGE, None)

fig, axes = plt.subplots(1, 4, figsize=(18, 6))

x = np.arange(len(MODEL_DISPLAY))
w = 0.55

for ax_idx, task in enumerate(TASKS):
    ax = axes[ax_idx]
    r  = all_results.get(task, {})

    aurocs, lo_errs, hi_errs, colors, hatches = [], [], [], [], []
    for m, (lbl, col, hatch) in MODEL_DISPLAY.items():
        if m not in r:
            aurocs.append(0); lo_errs.append(0); hi_errs.append(0)
            colors.append(col); hatches.append(hatch)
            continue
        aurocs.append(r[m]["auroc"])
        lo_errs.append(r[m]["auroc"] - r[m]["ci_lo"])
        hi_errs.append(r[m]["ci_hi"] - r[m]["auroc"])
        colors.append(col); hatches.append(hatch)

    bars = ax.bar(x, aurocs, width=w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for b, hatch in zip(bars, hatches):
        if hatch:
            b.set_hatch(hatch)
    ax.errorbar(x, aurocs, yerr=[lo_errs, hi_errs],
                fmt="none", color="#333333", capsize=4, linewidth=1.3)
    for xi, (a, he) in enumerate(zip(aurocs, hi_errs)):
        if a > 0:
            ax.text(xi, a + he + 0.004, f"{a:.3f}",
                    ha="center", fontsize=8, fontweight="bold", color="#333333")

    # MIMIC P reference line (ed_full equivalent from MIMIC)
    task_key = task if task in ("mortality", "readmission_30d") else None
    if task_key and "ed_full_behavioral" in mimic_p.get(task_key, {}):
        ref = mimic_p[task_key]["ed_full_behavioral"]["auroc"]
        ax.axhline(ref, color=BLUE, linewidth=1.5, linestyle="--", alpha=0.5)
        ax.text(len(x) - 0.3, ref + 0.003, f"MIMIC\n{ref:.3f}",
                fontsize=6.5, color=BLUE, ha="right", va="bottom")

    if aurocs and max(aurocs) > 0:
        ymin = max(0.50, min(a for a in aurocs if a > 0) - 0.06)
        ymax = max(aurocs) + max(hi_errs) + 0.06
        ax.set_ylim(ymin, min(ymax, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _, _ in MODEL_DISPLAY.values()],
                       fontsize=8, ha="center")
    ax.set_ylabel("AUROC" if ax_idx == 0 else "")
    ax.set_title(TASK_LABELS[task])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.text(0.98, 0.03,
            f"N={r.get('N_test', 0):,}\nPrev={r.get('prevalence_test', 0):.1%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#555555")

patches = [mpatches.Patch(color=c, alpha=0.85, label=l)
           for _, (l, c, _) in MODEL_DISPLAY.items()]
patches.append(Line2D([0], [0], color=BLUE, lw=1.5, ls="--", alpha=0.5,
                      label="MIMIC-IV reference (ed_full_behavioral)"))
fig.legend(handles=patches, loc="lower center", ncol=5,
           bbox_to_anchor=(0.5, -0.10), frameon=False, fontsize=8.5)

plt.tight_layout()
plt.savefig(FIG / "fig16_mcmed_validation.png", bbox_inches="tight")
plt.close()
print("Saved → fig16_mcmed_validation.png")

# ── 10. Print summary ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MC-MED EXTERNAL VALIDATION — SUMMARY")
print("=" * 70)
for task in TASKS:
    r = all_results.get(task, {})
    print(f"\n  {task}  (N={r.get('N_test', 0):,}, prev={r.get('prevalence_test', 0):.1%})")
    for m in MODEL_DISPLAY:
        if m in r:
            v = r[m]
            print(f"    {m:<25}: {v['auroc']:.4f} [{v['ci_lo']:.4f}–{v['ci_hi']:.4f}]")
print("=" * 70)
