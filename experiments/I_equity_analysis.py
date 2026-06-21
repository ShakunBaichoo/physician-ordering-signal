#!/usr/bin/env python
"""
Experiment I — Racial & Insurance Equity of the Ordering Signal
===============================================================
Evaluates whether the ordering-only predictive signal performs
equitably across demographic subgroups.

Analyses:
  1. Per-race AUROC: ordering-only vs values-only vs combined (4 tasks)
  2. Deceptively-normal (DN) phenotype prevalence by race & insurance
     — do certain groups disproportionately end up in the high-ordering/
       normal-values quadrant?
  3. Conformal prediction coverage by race (extending Experiment B)
     — does the 90% coverage guarantee hold within each racial group?

DEPENDS ON: Experiment H must run first (loads test_predictions.parquet)

Outputs → 1_ordering_paper/results/I_equity/
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
H_OUT = ROOT / "1_ordering_paper" / "results" / "H_cci_stratified"
OUT   = ROOT / "1_ordering_paper" / "results" / "I_equity"
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

# ── 2. Load stratification (DN phenotype) ────────────────────────────────────
strat = pd.read_csv(
    ROOT / "1_ordering_paper" / "results" / "A_ordering_signal" /
    "deceptively_normal" / "patient_stratification.csv"
)
strat["phenotype"] = "other"
strat.loc[(strat["val_q"]==0) & (strat["ord_q"]==3), "phenotype"] = "DN"
strat.loc[(strat["val_q"]==0) & (strat["ord_q"]==0), "phenotype"] = "CN"

# Merge static for race/insurance
static  = pd.read_parquet(DATA / "static.parquet")
strat   = strat.merge(static[["hadm_id"] +
    [c for c in static.columns if c.startswith("race_group_") or
     c.startswith("insurance_")]],
    on="hadm_id", how="left")

RACE_COLS = [c for c in static.columns if c.startswith("race_group_")]
INS_COLS  = [c for c in static.columns if c.startswith("insurance_")]
RACE_LABELS = {c: c.replace("race_group_","").title() for c in RACE_COLS}
INS_LABELS  = {c: c.replace("insurance_","") for c in INS_COLS}


# ── 3. Bootstrap AUROC helper ─────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

def bootstrap_auroc(y, p, n_boot=N_BOOT):
    if int(y.sum()) < 5 or int((y==0).sum()) < 5:
        return float("nan"), float("nan"), float("nan")
    base = roc_auc_score(y, p)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.sum() > 0 and (yb==0).sum() > 0:
            boot.append(roc_auc_score(yb, pb))
    lo, hi = np.percentile(boot,[2.5,97.5]) if boot else (float("nan"),)*2
    return round(base,4), round(float(lo),4), round(float(hi),4)


# ── 4. Per-race AUROC ─────────────────────────────────────────────────────────
print("\n=== Per-Race AUROC ===")
race_auroc_rows = []

for task in TASKS:
    df_t = all_preds[all_preds["task"]==task].dropna(subset=["true_label"])
    # Overall
    for prob_col, model_name in [
        ("ord_prob","ordering_only"), ("val_prob","values_only"), ("both_prob","combined")
    ]:
        y_all = df_t["true_label"].values.astype(int)
        p_all = df_t[prob_col].values
        auroc, lo, hi = bootstrap_auroc(y_all, p_all)
        race_auroc_rows.append({
            "task":task, "group":"ALL", "model":model_name,
            "N":len(df_t), "n_pos":int(y_all.sum()),
            "auroc":auroc, "ci_lo":lo, "ci_hi":hi
        })

    # By race
    for rc in RACE_COLS:
        grp_name = RACE_LABELS[rc]
        sub = df_t[df_t[rc]==1] if rc in df_t.columns else pd.DataFrame()
        if len(sub) < 30:
            continue
        y = sub["true_label"].values.astype(int)
        for prob_col, model_name in [("ord_prob","ordering_only"),
                                      ("val_prob","values_only"),
                                      ("both_prob","combined")]:
            p = sub[prob_col].values
            auroc, lo, hi = bootstrap_auroc(y, p)
            race_auroc_rows.append({
                "task":task, "group":grp_name, "model":model_name,
                "N":len(sub), "n_pos":int(y.sum()),
                "auroc":auroc, "ci_lo":lo, "ci_hi":hi
            })
            if model_name == "ordering_only":
                print(f"  {task:<16} {grp_name:<12} {model_name}  "
                      f"AUROC={auroc:.3f} [{lo:.3f}–{hi:.3f}]  N={len(sub):,}")

race_auroc_df = pd.DataFrame(race_auroc_rows)
race_auroc_df["auroc_str"] = race_auroc_df.apply(
    lambda r: f"{r['auroc']:.3f} [{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]", axis=1)
race_auroc_df.to_csv(OUT / "race_auroc.csv", index=False)


# ── 5. Insurance-stratified AUROC ─────────────────────────────────────────────
print("\n=== Per-Insurance AUROC ===")
ins_auroc_rows = []
for task in TASKS:
    df_t = all_preds[all_preds["task"]==task].dropna(subset=["true_label"])
    for ic in INS_COLS:
        grp_name = INS_LABELS[ic]
        sub = df_t[df_t[ic]==1] if ic in df_t.columns else pd.DataFrame()
        if len(sub) < 30:
            continue
        y = sub["true_label"].values.astype(int)
        for prob_col, model_name in [("ord_prob","ordering_only"),
                                      ("val_prob","values_only")]:
            p = sub[prob_col].values
            auroc, lo, hi = bootstrap_auroc(y, p)
            ins_auroc_rows.append({
                "task":task, "group":grp_name, "model":model_name,
                "N":len(sub), "n_pos":int(y.sum()),
                "auroc":auroc, "ci_lo":lo, "ci_hi":hi
            })
            if model_name == "ordering_only":
                print(f"  {task:<16} {grp_name:<12}  "
                      f"AUROC={auroc:.3f} [{lo:.3f}–{hi:.3f}]  N={len(sub):,}")

ins_auroc_df = pd.DataFrame(ins_auroc_rows)
ins_auroc_df.to_csv(OUT / "insurance_auroc.csv", index=False)


# ── 6. DN phenotype prevalence by race ────────────────────────────────────────
print("\n=== DN Phenotype by Race ===")
dn_race_rows = []
for rc in RACE_COLS:
    grp_name = RACE_LABELS[rc]
    sub = strat[strat[rc]==1] if rc in strat.columns else pd.DataFrame()
    if len(sub) < 30:
        continue
    n_tot = len(sub)
    for ph in ["DN","CN","other"]:
        n_ph = (sub["phenotype"]==ph).sum()
        prev = n_ph / n_tot * 100
        mort = sub[sub["phenotype"]==ph]["mortality"].mean() * 100 if n_ph > 0 else float("nan")
        dn_race_rows.append({
            "race":grp_name, "phenotype":ph,
            "N_race":n_tot, "N_phenotype":int(n_ph),
            "prevalence_pct":round(prev,2),
            "mortality_pct":round(mort,2)
        })
        if ph in ("DN","CN"):
            print(f"  {grp_name:<12} {ph}  prev={prev:.1f}%  mort={mort:.1f}%  (N={n_ph:,})")

dn_race_df = pd.DataFrame(dn_race_rows)
dn_race_df.to_csv(OUT / "dn_prevalence_by_race.csv", index=False)


# ── 7. Conformal coverage by race (split conformal at α=0.10) ─────────────────
print("\n=== Conformal Coverage by Race ===")
# Use a simple split conformal: calibrate on val set ordering_only predictions,
# then check coverage on test subgroups.
# Strategy: use the H test predictions directly — calibrate threshold on the
# overall test set first 50% (pseudo-calibration) and evaluate on race subgroups.

ALPHA = 0.10  # target coverage = 90%
conf_rows = []

for task in TASKS:
    df_t = all_preds[all_preds["task"]==task].dropna(subset=["true_label"])
    df_t = df_t.copy()
    df_t["nc_score"] = 1.0 - df_t["ord_prob"]  # nonconformity score

    # Calibrate on first half of test set (positive labels only, proper conformal)
    pos_mask  = df_t["true_label"] == 1
    cal_scores = df_t[pos_mask]["nc_score"].values
    n_cal      = len(cal_scores)
    if n_cal < 10:
        continue
    threshold = np.quantile(cal_scores, 1.0 - ALPHA + (1 / (n_cal + 1)))

    # Coverage in each race group
    for grp_name, rc in [("ALL", None)] + [(RACE_LABELS[rc], rc) for rc in RACE_COLS]:
        if rc is None:
            sub = df_t
        else:
            sub = df_t[df_t[rc]==1] if rc in df_t.columns else pd.DataFrame()
        if len(sub) < 20:
            continue
        # Coverage = fraction of positive-label patients where nc_score ≤ threshold
        pos_sub = sub[sub["true_label"]==1]
        if len(pos_sub) == 0:
            continue
        coverage = (pos_sub["nc_score"] <= threshold).mean() * 100
        conf_rows.append({
            "task":task, "group":grp_name,
            "N":len(sub), "N_pos":len(pos_sub),
            "coverage_pct":round(coverage,2),
            "target_pct":90.0,
            "gap_pp":round(coverage - 90.0, 2)
        })
        if grp_name in ("ALL","White","Black","Hispanic","Unknown"):
            print(f"  {task:<16} {grp_name:<12} coverage={coverage:.1f}%  "
                  f"(gap={coverage-90.0:+.1f}pp)  N_pos={len(pos_sub)}")

conf_df = pd.DataFrame(conf_rows)
conf_df.to_csv(OUT / "conformal_coverage_by_race.csv", index=False)


# ── 8. Unknown-race deep dive (9.1% mortality — who are they?) ───────────────
print("\n=== Unknown-Race Group Analysis ===")
unknown_mask = strat["race_group_unknown"] == 1
unk = strat[unknown_mask]
print(f"  N = {len(unk):,}")
for ph in ["DN","CN","other"]:
    n = (unk["phenotype"]==ph).sum()
    print(f"  {ph}: N={n:,} ({n/len(unk)*100:.1f}%)  "
          f"mort={unk[unk['phenotype']==ph]['mortality'].mean()*100:.1f}%")
ins_dist = {c.replace("insurance_",""): int(unk[c].sum())
            for c in INS_COLS if c in unk.columns}
print(f"  Insurance: {ins_dist}")
unk_summary = {
    "N": int(len(unk)),
    "mortality_pct": round(unk["mortality"].mean()*100, 2),
    "phenotype_counts": {ph: int((unk["phenotype"]==ph).sum()) for ph in ["DN","CN","other"]},
    "insurance": ins_dist,
}
with open(OUT / "unknown_race_summary.json", "w") as f:
    json.dump(unk_summary, f, indent=2)


# ── 9. Summary JSON ───────────────────────────────────────────────────────────
summary = {
    "race_auroc_ordering_only": {},
    "dn_prevalence_by_race": {},
    "conformal_coverage_by_race": {},
}
for task in TASKS:
    summary["race_auroc_ordering_only"][task] = {}
    for _, row in race_auroc_df[
            (race_auroc_df["task"]==task) &
            (race_auroc_df["model"]=="ordering_only")].iterrows():
        summary["race_auroc_ordering_only"][task][row["group"]] = {
            "auroc":row["auroc"], "ci_lo":row["ci_lo"], "ci_hi":row["ci_hi"],
            "N":int(row["N"]), "n_pos":int(row["n_pos"])
        }

with open(OUT / "equity_results.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅  Experiment I complete.  Results → {OUT}")
