#!/usr/bin/env python
"""
N_temporal_validation.py
=========================
Tests whether the ordering signal is temporally stable across real-world
year groups, addressing the concern that EHR ordering practices may have
shifted over the 11-year MIMIC-IV span.

Key clinical context:
  - Sepsis-3 definition published 2016 → altered sepsis recognition ordering
  - KDIGO AKI definition updated 2012 → altered creatinine monitoring
  - EHR adoption increased throughout 2008-2019 → ordering workflow changes

Split by anchor_year_group (real-world years, privacy-shifted in MIMIC):
  Train:  2008-2010 + 2011-2013  (~210K admissions)
  Val:    2014-2016              (~56K admissions)
  Test:   2017-2019              (~40K admissions, most temporally distant)

Same admission can appear in train and test if admitted across periods
(standard temporal validation: model trained on history, tested on future).

Outputs → 1_ordering_paper/results/N_temporal/
  temporal_results.json         — AUROC comparison vs random-split
  temporal_auroc.csv            — per-task/model results
  fig13_temporal.png            — 4-panel figure
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "1_ordering_paper" / "results" / "N_temporal"
FIG  = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# Load random-split baseline (from Experiment F bootstrap CIs)
RANDOM_CI_PATH = ROOT / "1_ordering_paper" / "results" / "F_paper_improvements" / "bootstrap_ci_results.json"

TASKS  = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED   = 42
N_BOOT = 1000

TRAIN_GROUPS = {"2008 - 2010", "2011 - 2013"}
VAL_GROUPS   = {"2014 - 2016"}
TEST_GROUPS  = {"2017 - 2019"}


# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading data...")
cohort = pd.read_parquet(DATA / "cohort.parquet")[
    ["hadm_id", "subject_id", "anchor_year_group"]
]
labels = pd.read_parquet(DATA / "labels.parquet")
static = pd.read_parquet(DATA / "static.parquet")

print("  Loading labs timeseries...")
labs_long   = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
print("  Loading vitals timeseries...")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

# Year-group distribution
yr_counts = cohort["anchor_year_group"].value_counts().sort_index()
print(f"\nYear-group distribution in cohort:")
for yr, n in yr_counts.items():
    split = ("TRAIN" if yr in TRAIN_GROUPS else
             "VAL"   if yr in VAL_GROUPS   else
             "TEST"  if yr in TEST_GROUPS  else "skip")
    print(f"  {yr}: {n:,}  [{split}]")


# ── 2. Feature extraction (identical to Experiment H) ─────────────────────────
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

    print(f"  Ordering features for {name} ({len(obs_cols)} tests)...")
    g = df_long.groupby("hadm_id", sort=False)

    # Per-test: total observations and intensity (fraction of bins ordered)
    total = g[obs_cols].sum()
    total.columns = [f"{c}_total" for c in obs_cols]

    n_bins = g["time_bin"].count()
    intensity = g[obs_cols].mean()
    intensity.columns = [f"{c}_intensity" for c in obs_cols]

    # Admission-level aggregates
    obs_data = df_long[["hadm_id"] + obs_cols].copy()
    obs_data["any_obs"] = (obs_data[obs_cols] > 0).any(axis=1).astype(int)
    adm_intensity = obs_data.groupby("hadm_id")["any_obs"].mean().rename("adm_ordering_intensity")

    # Diversity: mean number of distinct tests per bin
    obs_data["n_tests_bin"] = (obs_data[obs_cols] > 0).sum(axis=1)
    diversity = obs_data.groupby("hadm_id")["n_tests_bin"].mean().rename("adm_ordering_diversity")

    # Breadth: number of distinct tests ordered (ever)
    breadth = (g[obs_cols].sum() > 0).sum(axis=1).rename("adm_ordering_breadth")

    # Escalation: late_half ordering / early_half ordering
    n_bins_df = n_bins.rename("n_bins")
    mid = df_long.groupby("hadm_id")["time_bin"].transform("median")
    df_long["is_late"] = (df_long["time_bin"] >= mid).astype(int)
    early_int = (df_long[df_long["is_late"] == 0]
                 .groupby("hadm_id")[obs_cols].mean()
                 .mean(axis=1).rename("early_intensity"))
    late_int  = (df_long[df_long["is_late"] == 1]
                 .groupby("hadm_id")[obs_cols].mean()
                 .mean(axis=1).rename("late_intensity"))
    escalation = (late_int / (early_int + 1e-8)).rename("adm_ordering_escalation")
    df_long.drop(columns=["is_late"], inplace=True)

    agg = (total
           .join(intensity, how="left")
           .join(adm_intensity, how="left")
           .join(diversity, how="left")
           .join(breadth, how="left")
           .join(escalation, how="left"))
    agg.columns = [f"{c}__{name}" for c in agg.columns]
    return agg.reset_index()


print("\nExtracting value features...")
labs_val   = value_features(labs_long,   "lab")
vitals_val = value_features(vitals_long, "vit")

print("\nExtracting ordering features...")
labs_ord   = ordering_features(labs_long,   "lab")
vitals_ord = ordering_features(vitals_long, "vit")


# ── 3. Merge feature sets ──────────────────────────────────────────────────────
base = cohort.merge(labels, on=["hadm_id", "subject_id"])
META = ["hadm_id", "subject_id", "anchor_year_group"] + TASKS


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

feat_counts = {
    "values_only":   len([c for c in df_val.columns  if c not in META]),
    "ordering_only": len([c for c in df_ord.columns  if c not in META]),
    "combined":      len([c for c in df_both.columns if c not in META]),
}
print(f"\nFeature counts: {feat_counts}")


# ── 4. Temporal split (by anchor_year_group) ──────────────────────────────────
def temporal_split(df):
    tr = df["anchor_year_group"].isin(TRAIN_GROUPS)
    va = df["anchor_year_group"].isin(VAL_GROUPS)
    te = df["anchor_year_group"].isin(TEST_GROUPS)
    return tr, va, te


tr, va, te = temporal_split(df_both)
print(f"\nTemporal split — Train {tr.sum():,} | Val {va.sum():,} | Test {te.sum():,}")
print(f"  Test period: 2017-2019 (post-Sepsis-3, post-KDIGO)")

for task in TASKS:
    prev_tr  = df_both.loc[tr, task].mean()
    prev_te  = df_both.loc[te, task].mean()
    print(f"  {task}: train prev={prev_tr:.3f}  test prev={prev_te:.3f}")


# ── 5. LightGBM training ───────────────────────────────────────────────────────
LGBM_PARAMS = dict(
    n_estimators=2000, learning_rate=0.05, num_leaves=127,
    min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=-1, random_state=SEED, verbose=-1, metric="auc",
)


def train_model(df, task, tr_mask, va_mask):
    feat = [c for c in df.columns if c not in META]
    X_tr, y_tr = df.loc[tr_mask, feat].astype("float32"), df.loc[tr_mask, task]
    X_va, y_va = df.loc[va_mask, feat].astype("float32"), df.loc[va_mask, task]
    ok_tr = y_tr.notna()
    ok_va = y_va.notna()
    pos_w = (1 - y_tr[ok_tr].mean()) / y_tr[ok_tr].mean()
    m = lgb.LGBMClassifier(scale_pos_weight=pos_w, **LGBM_PARAMS)
    m.fit(X_tr[ok_tr], y_tr[ok_tr],
          eval_set=[(X_va[ok_va], y_va[ok_va])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    return m, feat


def evaluate(model, feat_cols, df, mask, task):
    X = df.loc[mask, feat_cols].astype("float32")
    y = df.loc[mask, task]
    ok = y.notna()
    probs = model.predict_proba(X[ok])[:, 1]
    return probs, y[ok].values


# Bootstrap CI
rng = np.random.default_rng(SEED)

def bootstrap_auroc(y, probs, n_boot=N_BOOT):
    n = len(y)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() < 2:
            continue
        scores.append(roc_auc_score(y[idx], probs[idx]))
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


print("\nTraining temporal models (this takes ~10–20 min)...")
results = {}

for task in TASKS:
    print(f"\n  {task}...")
    tr_t, va_t, te_t = temporal_split(df_both)  # same masks

    m_val,  f_val  = train_model(df_val,  task, tr_t, va_t)
    m_ord,  f_ord  = train_model(df_ord,  task, tr_t, va_t)
    m_both, f_both = train_model(df_both, task, tr_t, va_t)

    p_val,  y_val  = evaluate(m_val,  f_val,  df_val,  te_t, task)
    p_ord,  y_ord  = evaluate(m_ord,  f_ord,  df_ord,  te_t, task)
    p_both, y_both = evaluate(m_both, f_both, df_both, te_t, task)

    auc_v,  lo_v,  hi_v  = bootstrap_auroc(y_val,  p_val)
    auc_o,  lo_o,  hi_o  = bootstrap_auroc(y_ord,  p_ord)
    auc_b,  lo_b,  hi_b  = bootstrap_auroc(y_both, p_both)

    n_test = int(y_ord.shape[0])
    prev   = float(y_ord.mean())

    results[task] = {
        "N_test": n_test,
        "prevalence_test": round(prev, 4),
        "ordering_only": {"auroc": round(auc_o, 4), "ci_lo": round(lo_o, 4), "ci_hi": round(hi_o, 4)},
        "values_only":   {"auroc": round(auc_v, 4), "ci_lo": round(lo_v, 4), "ci_hi": round(hi_v, 4)},
        "combined":      {"auroc": round(auc_b, 4), "ci_lo": round(lo_b, 4), "ci_hi": round(hi_b, 4)},
    }

    print(f"    ordering_only:  AUROC {auc_o:.4f} [{lo_o:.4f}–{hi_o:.4f}]")
    print(f"    values_only:    AUROC {auc_v:.4f} [{lo_v:.4f}–{hi_v:.4f}]")
    print(f"    combined:       AUROC {auc_b:.4f} [{lo_b:.4f}–{hi_b:.4f}]")


# ── 6. Save results ────────────────────────────────────────────────────────────
with open(OUT / "temporal_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved temporal_results.json")

# Also save as CSV
rows = []
for task, r in results.items():
    for model in ["ordering_only", "values_only", "combined"]:
        rows.append({
            "task": task, "model": model,
            "auroc": r[model]["auroc"],
            "ci_lo": r[model]["ci_lo"],
            "ci_hi": r[model]["ci_hi"],
            "N_test": r["N_test"],
            "prevalence": r["prevalence_test"],
        })
pd.DataFrame(rows).to_csv(OUT / "temporal_auroc.csv", index=False)
print(f"Saved temporal_auroc.csv")


# ── 7. Compare to random-split baseline ────────────────────────────────────────
print("\n" + "="*70)
print("TEMPORAL VALIDATION — COMPARISON TO RANDOM-SPLIT BASELINE")
print("="*70)

try:
    with open(RANDOM_CI_PATH) as f:
        rand_ci = json.load(f)
    has_rand = True
except FileNotFoundError:
    has_rand = False
    print("  (random-split CI file not found — comparison skipped)")

for task in TASKS:
    r = results[task]
    print(f"\n  {task}  (temporal test N={r['N_test']:,}, prev={r['prevalence_test']:.1%})")
    for model in ["ordering_only", "values_only", "combined"]:
        auc = r[model]["auroc"]; lo = r[model]["ci_lo"]; hi = r[model]["ci_hi"]
        if has_rand and task in rand_ci and model in rand_ci[task]:
            r_auc = rand_ci[task][model]["auroc"]
            delta = auc - r_auc
            print(f"    {model:<16}: temporal {auc:.4f} [{lo:.4f}–{hi:.4f}]  "
                  f"vs random-split {r_auc:.4f}  Δ={delta:+.4f}")
        else:
            print(f"    {model:<16}: temporal {auc:.4f} [{lo:.4f}–{hi:.4f}]")

print("="*70)


# ── 8. Figure ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"

TASK_LABELS = {
    "mortality":       "In-Hospital Mortality",
    "readmission_30d": "30-Day Readmission",
    "aki":             "AKI",
    "sepsis":          "Sepsis",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

print("\nBuilding Figure 13 — Temporal Validation...")

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
fig.suptitle(
    "Figure 13 — Temporal Validation: Train on 2008–2013, Test on 2017–2019\n"
    "Assesses stability of ordering signal across evolving EHR practices "
    "(post-Sepsis-3 and post-KDIGO AKI guideline periods)",
    fontsize=11, fontweight="bold", y=1.03
)

MODELS = [
    ("ordering_only", "Ordering-only",  BLUE,   "///"),
    ("values_only",   "Values-only",    ORANGE, None),
    ("combined",      "Combined",       GREEN,  None),
]

for ax_idx, task in enumerate(TASKS):
    ax = axes[ax_idx]
    r  = results[task]
    x  = np.arange(len(MODELS))
    w  = 0.55

    aurocs = [r[m]["auroc"] for m, _, _, _ in MODELS]
    lo_err = [r[m]["auroc"] - r[m]["ci_lo"] for m, _, _, _ in MODELS]
    hi_err = [r[m]["ci_hi"] - r[m]["auroc"] for m, _, _, _ in MODELS]

    colors = [c for _, _, c, _ in MODELS]
    hatches = [h for _, _, _, h in MODELS]
    bars = ax.bar(x, aurocs, width=w, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for b, hatch in zip(bars, hatches):
        if hatch:
            b.set_hatch(hatch)
    ax.errorbar(x, aurocs, yerr=[lo_err, hi_err],
                fmt="none", color="#333333", capsize=4, linewidth=1.3)

    # Annotate values
    for xi, (a, he) in enumerate(zip(aurocs, hi_err)):
        ax.text(xi, a + he + 0.004, f"{a:.3f}",
                ha="center", fontsize=8, fontweight="bold", color="#333333")

    # Overlay random-split as dashed reference lines
    if has_rand and task in rand_ci:
        for mi, (model_name, label, color, _) in enumerate(MODELS):
            if model_name in rand_ci[task]:
                ref = rand_ci[task][model_name]["auroc"]
                ax.plot([mi - w/2 - 0.05, mi + w/2 + 0.05], [ref, ref],
                        color=color, linewidth=1.5, linestyle="--", alpha=0.7,
                        zorder=5)

    ymin = max(0.50, min(aurocs) - 0.06)
    ymax = max(aurocs) + max(hi_err) + 0.05
    ax.set_ylim(ymin, min(ymax + 0.01, 1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l, _, _ in MODELS], fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("AUROC" if ax_idx == 0 else "")
    ax.set_title(TASK_LABELS[task])
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    # Annotation: N and prevalence
    ax.text(0.98, 0.03,
            f"Test N={r['N_test']:,}\nPrev={r['prevalence_test']:.1%}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="#555555")

# Legend
patch_list = [mpatches.Patch(color=c, label=l, alpha=0.85)
              for _, l, c, _ in MODELS]
from matplotlib.lines import Line2D
patch_list.append(Line2D([0], [0], color="#888888", linewidth=1.5, linestyle="--",
                          label="Random-split reference"))
fig.legend(handles=patch_list, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.07), frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(FIG / "fig13_temporal.png", bbox_inches="tight")
plt.close()
print("  → fig13_temporal.png")

print(f"\nAll temporal validation outputs → {OUT}")
