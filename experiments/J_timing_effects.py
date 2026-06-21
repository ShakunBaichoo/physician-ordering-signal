#!/usr/bin/env python
"""
Experiment J — Admission Timing Effects on the Ordering Signal
==============================================================
Tests whether the ordering signal generalises across admission timing
contexts, where test-ordering patterns might plausibly differ.

Analyses:
  1. Night vs Day admission AUROC (ordering-only vs values-only vs combined)
     — do night-shift orderings carry the same predictive information?
  2. Weekend vs Weekday AUROC
     — does reduced staffing affect the signal quality?
  3. Hour-of-day AUROC across 6 four-hour blocks (0–4, 4–8, …, 20–24)
     — fine-grained time-of-day pattern
  4. Night × Weekend interaction (4 cells: night/day × weekend/weekday)
     — the most constrained context: Saturday night at 3am
  5. Ordering intensity descriptives by timing group
     — do physicians actually order differently at night / weekends?

DEPENDS ON: Experiment H must run first (loads test_predictions.parquet)

Outputs → 1_ordering_paper/results/J_timing/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT  = Path(__file__).parents[2]
DATA  = ROOT / "data" / "processed"
H_OUT = ROOT / "1_ordering_paper" / "results" / "H_cci_stratified"
OUT   = ROOT / "1_ordering_paper" / "results" / "J_timing"
OUT.mkdir(parents=True, exist_ok=True)

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED   = 42
N_BOOT = 1000

# ── 1. Load test predictions (built by Experiment H) ─────────────────────────
pred_path = H_OUT / "test_predictions.parquet"
if not pred_path.exists():
    raise FileNotFoundError(
        "Run H_cci_stratified.py first — test_predictions.parquet not found."
    )

print("Loading test predictions from Experiment H...")
all_preds = pd.read_parquet(pred_path)

# Check timing columns are present
timing_cols = ["admit_night", "admit_weekend", "admit_dow", "admit_hour"]
missing = [c for c in timing_cols if c not in all_preds.columns]
if missing:
    # Fall back: re-merge from static (handles old H runs without timing cols)
    print(f"  Timing columns missing from parquet ({missing}); merging from static...")
    static = pd.read_parquet(DATA / "static.parquet")
    all_preds = all_preds.merge(
        static[["hadm_id"] + [c for c in timing_cols if c in static.columns]],
        on="hadm_id", how="left"
    )

print(f"  Loaded {len(all_preds):,} rows × {all_preds.shape[1]} columns")
print(f"  Timing coverage: "
      f"night={all_preds['admit_night'].mean()*100:.1f}%  "
      f"weekend={all_preds['admit_weekend'].mean()*100:.1f}%")


# ── 2. Bootstrap AUROC helper ─────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def bootstrap_auroc(y, p, n_boot=N_BOOT):
    if int(y.sum()) < 5 or int((y == 0).sum()) < 5:
        return float("nan"), float("nan"), float("nan")
    base = roc_auc_score(y, p)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() > 0 and (yb == 0).sum() > 0:
            boot.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"),) * 2
    return round(base, 4), round(float(lo), 4), round(float(hi), 4)


# ── 3. Analysis 1: Night vs Day ───────────────────────────────────────────────
print("\n=== Night vs Day AUROC ===")
night_rows = []

for task in TASKS:
    df_t = all_preds[all_preds["task"] == task].dropna(subset=["true_label"])
    for label, mask_val in [("ALL", None), ("Day (06–22)", 0), ("Night (22–06)", 1)]:
        if mask_val is None:
            sub = df_t
        else:
            sub = df_t[df_t["admit_night"] == mask_val]
        if len(sub) < 30:
            continue
        y = sub["true_label"].values.astype(int)
        for prob_col, model_name in [
            ("ord_prob",  "ordering_only"),
            ("val_prob",  "values_only"),
            ("both_prob", "combined"),
        ]:
            p = sub[prob_col].values
            auroc, lo, hi = bootstrap_auroc(y, p)
            night_rows.append({
                "task": task, "group": label, "model": model_name,
                "N": len(sub), "n_pos": int(y.sum()),
                "auroc": auroc, "ci_lo": lo, "ci_hi": hi,
                "auroc_str": f"{auroc:.3f} [{lo:.3f}–{hi:.3f}]",
            })
        # Print ordering-only for readability
        r = night_rows[-3]  # most recent ordering_only row
        print(f"  {task:<16} {label:<20} "
              f"AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]  "
              f"N={len(sub):,}")

night_df = pd.DataFrame(night_rows)
night_df.to_csv(OUT / "night_day_auroc.csv", index=False)


# ── 4. Analysis 2: Weekend vs Weekday ────────────────────────────────────────
print("\n=== Weekend vs Weekday AUROC ===")
wknd_rows = []

for task in TASKS:
    df_t = all_preds[all_preds["task"] == task].dropna(subset=["true_label"])
    for label, mask_val in [("ALL", None), ("Weekday", 0), ("Weekend", 1)]:
        if mask_val is None:
            sub = df_t
        else:
            sub = df_t[df_t["admit_weekend"] == mask_val]
        if len(sub) < 30:
            continue
        y = sub["true_label"].values.astype(int)
        for prob_col, model_name in [
            ("ord_prob",  "ordering_only"),
            ("val_prob",  "values_only"),
            ("both_prob", "combined"),
        ]:
            p = sub[prob_col].values
            auroc, lo, hi = bootstrap_auroc(y, p)
            wknd_rows.append({
                "task": task, "group": label, "model": model_name,
                "N": len(sub), "n_pos": int(y.sum()),
                "auroc": auroc, "ci_lo": lo, "ci_hi": hi,
                "auroc_str": f"{auroc:.3f} [{lo:.3f}–{hi:.3f}]",
            })
        r = wknd_rows[-3]
        print(f"  {task:<16} {label:<12} "
              f"AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]  "
              f"N={len(sub):,}")

wknd_df = pd.DataFrame(wknd_rows)
wknd_df.to_csv(OUT / "weekend_weekday_auroc.csv", index=False)


# ── 5. Analysis 3: Hour-of-day blocks ─────────────────────────────────────────
print("\n=== Hour-of-Day Blocks ===")
HOUR_BLOCKS = [
    ("00–04", (0, 4)),
    ("04–08", (4, 8)),
    ("08–12", (8, 12)),
    ("12–16", (12, 16)),
    ("16–20", (16, 20)),
    ("20–24", (20, 24)),
]

hour_rows = []
for task in TASKS:
    df_t = all_preds[all_preds["task"] == task].dropna(subset=["true_label", "admit_hour"])
    for blk_name, (h_lo, h_hi) in HOUR_BLOCKS:
        sub = df_t[(df_t["admit_hour"] >= h_lo) & (df_t["admit_hour"] < h_hi)]
        if len(sub) < 30:
            continue
        y = sub["true_label"].values.astype(int)
        for prob_col, model_name in [("ord_prob","ordering_only"),
                                      ("val_prob","values_only")]:
            p = sub[prob_col].values
            auroc, lo, hi = bootstrap_auroc(y, p)
            hour_rows.append({
                "task": task, "hour_block": blk_name, "model": model_name,
                "N": len(sub), "n_pos": int(y.sum()),
                "auroc": auroc, "ci_lo": lo, "ci_hi": hi,
            })

hour_df = pd.DataFrame(hour_rows)
hour_df.to_csv(OUT / "hourly_auroc.csv", index=False)
# Summary print
pivot = hour_df[hour_df["model"]=="ordering_only"].pivot(
    index="hour_block", columns="task", values="auroc")
print(pivot.round(3).to_string())


# ── 6. Analysis 4: Night × Weekend interaction ────────────────────────────────
print("\n=== Night × Weekend Interaction ===")
interaction_rows = []

cells = [
    ("Day + Weekday",    {"admit_night": 0, "admit_weekend": 0}),
    ("Day + Weekend",    {"admit_night": 0, "admit_weekend": 1}),
    ("Night + Weekday",  {"admit_night": 1, "admit_weekend": 0}),
    ("Night + Weekend",  {"admit_night": 1, "admit_weekend": 1}),
]

for task in TASKS:
    df_t = all_preds[all_preds["task"] == task].dropna(subset=["true_label"])
    for cell_name, filters in cells:
        sub = df_t.copy()
        for col, val in filters.items():
            sub = sub[sub[col] == val]
        if len(sub) < 20:
            continue
        y = sub["true_label"].values.astype(int)
        for prob_col, model_name in [("ord_prob","ordering_only"),
                                      ("val_prob","values_only")]:
            p = sub[prob_col].values
            auroc, lo, hi = bootstrap_auroc(y, p)
            interaction_rows.append({
                "task": task, "cell": cell_name, "model": model_name,
                "N": len(sub), "n_pos": int(y.sum()),
                "auroc": auroc, "ci_lo": lo, "ci_hi": hi,
                "auroc_str": f"{auroc:.3f} [{lo:.3f}–{hi:.3f}]",
            })
        r = interaction_rows[-2]  # ordering_only row
        print(f"  {task:<16} {cell_name:<18} "
              f"AUROC={r['auroc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]  "
              f"N={len(sub):,}")

interaction_df = pd.DataFrame(interaction_rows)
interaction_df.to_csv(OUT / "night_weekend_interaction.csv", index=False)


# ── 7. Analysis 5: Ordering intensity by timing group ────────────────────────
print("\n=== Ordering Intensity by Timing Group ===")
# Re-load ordering features from static+df_ord if available;
# otherwise use the columns already in all_preds (none there — compute from A output)
# Strategy: load patient_stratification.csv (has ordering_intensity from Exp A)
strat_path = (ROOT / "1_ordering_paper" / "results" / "A_ordering_signal" /
               "deceptively_normal" / "patient_stratification.csv")
if strat_path.exists():
    strat = pd.read_csv(strat_path)
    strat["phenotype"] = "other"
    strat.loc[(strat["val_q"]==0) & (strat["ord_q"]==3), "phenotype"] = "DN"
    strat.loc[(strat["val_q"]==0) & (strat["ord_q"]==0), "phenotype"] = "CN"
    # Merge timing onto strat
    timing_sub = (all_preds[all_preds["task"] == "mortality"]
                  [["hadm_id","admit_night","admit_weekend","admit_hour"]]
                  .drop_duplicates("hadm_id"))
    strat_t = strat.merge(timing_sub, on="hadm_id", how="inner")

    intensity_rows = []
    for timing_var, label_map in [
        ("admit_night",   {0: "Day", 1: "Night"}),
        ("admit_weekend", {0: "Weekday", 1: "Weekend"}),
    ]:
        for val, lbl in label_map.items():
            sub = strat_t[strat_t[timing_var] == val]
            if len(sub) == 0:
                continue
            intensity_rows.append({
                "timing_var": timing_var,
                "group": lbl,
                "N": len(sub),
                "ord_intensity_mean": round(sub["ordering_intensity_overall"].mean(), 4),
                "ord_intensity_std":  round(sub["ordering_intensity_overall"].std(), 4),
                "ord_q_mean":         round(sub["ord_q"].mean(), 3),
                "val_q_mean":         round(sub["val_q"].mean(), 3),
                "dn_pct":             round((sub["phenotype"]=="DN").mean()*100
                                             if "phenotype" in sub.columns else float("nan"), 2),
            })
            print(f"  {timing_var:<16} {lbl:<10}  "
                  f"ord_intensity={sub['ordering_intensity_overall'].mean():.4f}  "
                  f"ord_q={sub['ord_q'].mean():.2f}  N={len(sub):,}")

    intensity_df = pd.DataFrame(intensity_rows)
    intensity_df.to_csv(OUT / "ordering_intensity_by_timing.csv", index=False)
else:
    print("  patient_stratification.csv not found — skipping intensity descriptives.")
    intensity_df = pd.DataFrame()


# ── 8. Summary JSON ───────────────────────────────────────────────────────────
summary = {
    "night_day": {},
    "weekend_weekday": {},
    "interaction": {},
}

for task in TASKS:
    summary["night_day"][task] = {}
    for _, row in night_df[
            (night_df["task"] == task) &
            (night_df["model"] == "ordering_only")].iterrows():
        summary["night_day"][task][row["group"]] = {
            "auroc": row["auroc"], "ci_lo": row["ci_lo"],
            "ci_hi": row["ci_hi"], "N": int(row["N"]), "n_pos": int(row["n_pos"])
        }

    summary["weekend_weekday"][task] = {}
    for _, row in wknd_df[
            (wknd_df["task"] == task) &
            (wknd_df["model"] == "ordering_only")].iterrows():
        summary["weekend_weekday"][task][row["group"]] = {
            "auroc": row["auroc"], "ci_lo": row["ci_lo"],
            "ci_hi": row["ci_hi"], "N": int(row["N"]), "n_pos": int(row["n_pos"])
        }

    summary["interaction"][task] = {}
    for _, row in interaction_df[
            (interaction_df["task"] == task) &
            (interaction_df["model"] == "ordering_only")].iterrows():
        summary["interaction"][task][row["cell"]] = {
            "auroc": row["auroc"], "ci_lo": row["ci_lo"],
            "ci_hi": row["ci_hi"], "N": int(row["N"]), "n_pos": int(row["n_pos"])
        }

with open(OUT / "timing_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅  Experiment J complete.  Results → {OUT}")
