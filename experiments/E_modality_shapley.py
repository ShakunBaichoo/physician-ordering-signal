#!/usr/bin/env python
"""
Experiment E — Coalition Game: Shapley Value of Each Clinical Data Stream
=========================================================================
Treats each clinical data modality as a "player" in a cooperative game.
The value of each coalition = AUROC achieved by training LightGBM on that
subset of modalities.

Players (modalities):
  S  — static features (demographics, comorbidities, medications)
  L  — lab values (aggregated across 48h)
  V  — vital signs (aggregated across 48h)
  O  — ordering patterns (physician test-ordering frequencies)

With 4 players there are 2^4 = 16 coalitions (including empty set).
We train one LightGBM per coalition × task = 60 models (excluding empty).

Shapley value of player i:
  φ_i = Σ_{S ⊆ N\{i}} [|S|!(|N|-|S|-1)!/|N|!] * [v(S∪{i}) - v(S)]

This gives a fair, axiomatic attribution of each modality's marginal
contribution to predictive performance — a unique game-theoretic answer to:
"What is each clinical data stream actually worth?"

Outputs (results/novel/E_modality_shapley/):
  - coalition_aurocs.csv       AUROC for each of 15 coalitions × 4 tasks
  - shapley_values.csv         Shapley φ per modality per task
  - marginal_contributions.csv pairwise marginal gains
"""

import json
import math
import warnings
from itertools import combinations, chain
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "1_ordering_paper" / "results" / "E_modality_shapley"
OUT.mkdir(parents=True, exist_ok=True)

TASKS    = ["mortality", "readmission_30d", "aki", "sepsis"]
PLAYERS  = ["S", "L", "V", "O"]   # Static, Labs, Vitals, Ordering
SEED     = 42


# ── 1. Build modality feature sets ────────────────────────────────────────────
print("Loading data...")
cohort  = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
labels  = pd.read_parquet(DATA / "labels.parquet")
static  = pd.read_parquet(DATA / "static.parquet")

print("  Labs...")
labs_long = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
print("  Vitals...")
vitals_long = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

def agg_val(df_long, suffix):
    val_cols = [c for c in df_long.columns
                if c not in ("hadm_id","time_bin") and not c.endswith("_obs")]
    agg = df_long.groupby("hadm_id", sort=False)[val_cols].mean()
    agg.columns = [f"{c}__{suffix}" for c in agg.columns]
    return agg.reset_index()

def agg_ord(df_long, suffix):
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    total = df_long.groupby("hadm_id", sort=False)[obs_cols].sum()
    total.columns = [f"{c}_total__{suffix}" for c in obs_cols]
    inten = df_long.groupby("hadm_id", sort=False)[obs_cols].apply(
        lambda g: pd.Series({"ordering_intensity": (g > 0).values.mean()}))
    return total.join(inten).reset_index()

labs_val   = agg_val(labs_long,   "L")
vitals_val = agg_val(vitals_long, "V")
labs_ord   = agg_ord(labs_long,   "O_lab")
vitals_ord = agg_ord(vitals_long, "O_vit")

# Map player → feature DataFrame
base = cohort.merge(labels, on=["hadm_id","subject_id"])
modality_dfs = {
    "S": static,
    "L": labs_val,
    "V": vitals_val,
    "O": labs_ord.merge(vitals_ord, on="hadm_id", how="outer"),
}

meta_cols = ["hadm_id", "subject_id"] + TASKS


# ── 2. Patient split ──────────────────────────────────────────────────────────
# Build full df just to get consistent subject split
full = base.copy()
for mdf in modality_dfs.values():
    full = full.merge(mdf, on="hadm_id", how="left")

pat = full.groupby("subject_id")["mortality"].max().reset_index()
pat = pat.sample(frac=1, random_state=SEED)
n = len(pat)
n_train, n_val = int(0.70 * n), int(0.15 * n)
train_s = set(pat.iloc[:n_train]["subject_id"])
val_s   = set(pat.iloc[n_train:n_train+n_val]["subject_id"])
test_s  = set(pat.iloc[n_train+n_val:]["subject_id"])

tr_m = full["subject_id"].isin(train_s)
va_m = full["subject_id"].isin(val_s)
te_m = full["subject_id"].isin(test_s)

print(f"Split — Train {tr_m.sum():,} | Val {va_m.sum():,} | Test {te_m.sum():,}")


# ── 3. Enumerate all non-empty coalitions ────────────────────────────────────
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

coalitions = list(powerset(PLAYERS))
print(f"\n{len(coalitions)} coalitions to evaluate × {len(TASKS)} tasks "
      f"= {len(coalitions)*len(TASKS)} models\n")


# ── 4. Train & evaluate each coalition ───────────────────────────────────────
def build_coalition_df(coalition):
    df = base.copy()
    for player in coalition:
        df = df.merge(modality_dfs[player], on="hadm_id", how="left")
    return df

def train_coalition(coalition, task):
    df = build_coalition_df(coalition)
    feat_cols = [c for c in df.columns if c not in meta_cols]
    if not feat_cols:
        return np.nan

    y_tr = df.loc[tr_m, task]; y_va = df.loc[va_m, task]; y_te = df.loc[te_m, task]
    ok_tr, ok_va, ok_te = y_tr.notna(), y_va.notna(), y_te.notna()

    X_tr = df.loc[tr_m, feat_cols].astype("float32")
    X_va = df.loc[va_m, feat_cols].astype("float32")
    X_te = df.loc[te_m, feat_cols].astype("float32")

    prev = y_tr[ok_tr].mean()
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=(1-prev)/prev,
        n_jobs=-1, random_state=SEED, verbose=-1, metric="auc",
    )
    model.fit(X_tr[ok_tr], y_tr[ok_tr],
              eval_set=[(X_va[ok_va], y_va[ok_va])],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])

    prob = model.predict_proba(X_te[ok_te])[:, 1]
    return roc_auc_score(y_te[ok_te], prob)

# Results dict: coalition_key → {task → auroc}
coalition_aurocs = {}

for i, coalition in enumerate(coalitions):
    key = "+".join(coalition)
    coalition_aurocs[key] = {}
    print(f"[{i+1:2d}/{len(coalitions)}] Coalition {key:<10}", end="  ")
    for task in TASKS:
        auroc = train_coalition(coalition, task)
        coalition_aurocs[key][task] = round(float(auroc), 4)
        print(f"{task[:4]}={auroc:.3f}", end="  ")
    print()

# Save raw coalition AUROC table
rows = []
for key, task_aurocs in coalition_aurocs.items():
    row = {"coalition": key, "n_players": len(key.split("+"))}
    row.update(task_aurocs)
    rows.append(row)
coal_df = pd.DataFrame(rows).sort_values("n_players")
coal_df.to_csv(OUT / "coalition_aurocs.csv", index=False)


# ── 5. Compute Shapley values ─────────────────────────────────────────────────
def v(coalition_tuple, task):
    """Value function = AUROC for this coalition."""
    if not coalition_tuple:
        return 0.5   # random baseline (AUROC of random classifier)
    key = "+".join(sorted(coalition_tuple))
    # Try all orderings of the key
    for k in coalition_aurocs:
        if set(k.split("+")) == set(coalition_tuple):
            return coalition_aurocs[k][task]
    return 0.5

def shapley(player, all_players, task):
    """Exact Shapley value for player given value function v."""
    n = len(all_players)
    others = [p for p in all_players if p != player]
    phi = 0.0
    for r in range(len(others) + 1):
        for S in combinations(others, r):
            S = list(S)
            weight = (math.factorial(len(S)) *
                      math.factorial(n - len(S) - 1) /
                      math.factorial(n))
            marginal = v(tuple(S + [player]), task) - v(tuple(S), task)
            phi += weight * marginal
    return phi

print("\nComputing Shapley values...")
shapley_rows = []
for task in TASKS:
    for player in PLAYERS:
        phi = shapley(player, PLAYERS, task)
        shapley_rows.append({"task": task, "player": player,
                             "shapley_value": round(phi, 5)})
        print(f"  φ({player}) task={task:<18} = {phi:+.5f}")

shap_df = pd.DataFrame(shapley_rows)
shap_df.to_csv(OUT / "shapley_values.csv", index=False)

# Pivot table for easy reading
pivot = shap_df.pivot(index="player", columns="task", values="shapley_value")
print(f"\n{'='*60}")
print("  SHAPLEY VALUES (marginal AUROC contribution)")
print(f"{'='*60}")
print(pivot.to_string())


# ── 6. Marginal contributions (pairwise) ─────────────────────────────────────
print("\nMarginal value of Ordering (O) given other modalities:")
marginal_rows = []
for task in TASKS:
    for base_coal in powerset([p for p in PLAYERS if p != "O"]):
        base_coal = list(base_coal)
        with_O    = base_coal + ["O"]
        gain = v(tuple(with_O), task) - v(tuple(base_coal), task)
        marginal_rows.append({
            "task": task,
            "base_coalition": "+".join(base_coal),
            "auroc_without_O": v(tuple(base_coal), task),
            "auroc_with_O":    v(tuple(with_O), task),
            "marginal_gain_O": round(gain, 5),
        })

marg_df = pd.DataFrame(marginal_rows)
marg_df.to_csv(OUT / "marginal_contributions_O.csv", index=False)

print(marg_df[["task","base_coalition","marginal_gain_O"]].to_string(index=False))

# Summary
with open(OUT / "game_theory_summary.json", "w") as f:
    json.dump({
        "n_coalitions": len(coalitions),
        "n_tasks": len(TASKS),
        "shapley_values": shap_df.to_dict(orient="records"),
        "coalition_aurocs": coalition_aurocs,
    }, f, indent=2)

print(f"\nAll results saved to {OUT}/")
print("  coalition_aurocs.csv          — AUROC for all 15 coalitions")
print("  shapley_values.csv            — Shapley φ per modality per task")
print("  marginal_contributions_O.csv  — marginal gain of ordering data")
print("  game_theory_summary.json      — complete results")
