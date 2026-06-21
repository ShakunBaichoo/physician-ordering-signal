#!/usr/bin/env python
"""Generate paired/bootstrap statistical comparisons from saved predictions.

This script covers comparisons where row-level predictions are already saved:
MIMIC test-set model comparisons, timing subgroup differences, and race subgroup
heterogeneity for the ordering-only model.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parents[2]
PAPER = ROOT / "1_ordering_paper"
PRED = PAPER / "results" / "H_cci_stratified" / "test_predictions.parquet"
OUT = PAPER / "results" / "paired_statistical_comparisons"
OUT.mkdir(parents=True, exist_ok=True)

N_BOOT = 2000
SEED = 42
TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]


def paired_bootstrap_auc_diff(y, a, b, n_boot=N_BOOT, seed=SEED):
    y = np.asarray(y)
    a = np.asarray(a)
    b = np.asarray(b)
    obs = roc_auc_score(y, a) - roc_auc_score(y, b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        diffs.append(roc_auc_score(yb, a[idx]) - roc_auc_score(yb, b[idx]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return obs, lo, hi, min(float(p), 1.0)


def independent_bootstrap_auc_diff(y_a, p_a, y_b, p_b, n_boot=N_BOOT, seed=SEED):
    y_a = np.asarray(y_a)
    p_a = np.asarray(p_a)
    y_b = np.asarray(y_b)
    p_b = np.asarray(p_b)
    obs = roc_auc_score(y_a, p_a) - roc_auc_score(y_b, p_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        ia = rng.integers(0, len(y_a), len(y_a))
        ib = rng.integers(0, len(y_b), len(y_b))
        ya = y_a[ia]
        yb = y_b[ib]
        if ya.sum() == 0 or ya.sum() == len(ya) or yb.sum() == 0 or yb.sum() == len(yb):
            continue
        diffs.append(roc_auc_score(ya, p_a[ia]) - roc_auc_score(yb, p_b[ib]))
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    return obs, lo, hi, min(float(p), 1.0)


def race_group(row):
    for label, col in [
        ("Asian", "race_group_asian"),
        ("Black", "race_group_black"),
        ("Hispanic", "race_group_hispanic"),
        ("Native", "race_group_native"),
        ("Other", "race_group_other"),
        ("Unknown", "race_group_unknown"),
        ("White", "race_group_white"),
    ]:
        if row.get(col, 0) == 1:
            return label
    return "Unknown"


df = pd.read_parquet(PRED)
df = df[df["task"].isin(TASKS)].copy()
df["race_group"] = df.apply(race_group, axis=1)

model_map = {
    "values_only": "val_prob",
    "ordering_only": "ord_prob",
    "combined": "both_prob",
}

model_rows = []
for task in TASKS:
    sub = df[df["task"] == task].dropna(subset=["true_label", "val_prob", "ord_prob", "both_prob"])
    comparisons = [
        ("ordering_only", "values_only"),
        ("combined", "ordering_only"),
        ("combined", "values_only"),
    ]
    for a, b in comparisons:
        diff, lo, hi, p = paired_bootstrap_auc_diff(
            sub["true_label"].values, sub[model_map[a]].values, sub[model_map[b]].values
        )
        model_rows.append({
            "dataset": "MIMIC-IV test",
            "task": task,
            "model_a": a,
            "model_b": b,
            "comparison": f"{a} - {b}",
            "n": int(len(sub)),
            "n_pos": int(sub["true_label"].sum()),
            "auc_a": roc_auc_score(sub["true_label"], sub[model_map[a]]),
            "auc_b": roc_auc_score(sub["true_label"], sub[model_map[b]]),
            "auc_diff": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "p_value_two_sided": p,
            "test_type": "paired bootstrap on same admissions",
        })

timing_rows = []
timing_specs = [
    ("night_vs_day", "admit_night", 1, 0, "Night (22-06)", "Day (06-22)"),
    ("weekend_vs_weekday", "admit_weekend", 1, 0, "Weekend", "Weekday"),
]
for task in TASKS:
    task_df = df[df["task"] == task].dropna(subset=["true_label", "ord_prob"])
    for comparison_name, col, a_val, b_val, a_label, b_label in timing_specs:
        a = task_df[task_df[col] == a_val]
        b = task_df[task_df[col] == b_val]
        diff, lo, hi, p = independent_bootstrap_auc_diff(
            a["true_label"].values, a["ord_prob"].values,
            b["true_label"].values, b["ord_prob"].values,
        )
        timing_rows.append({
            "dataset": "MIMIC-IV test",
            "task": task,
            "model": "ordering_only",
            "comparison": comparison_name,
            "group_a": a_label,
            "group_b": b_label,
            "n_a": int(len(a)),
            "n_pos_a": int(a["true_label"].sum()),
            "n_b": int(len(b)),
            "n_pos_b": int(b["true_label"].sum()),
            "auc_a": roc_auc_score(a["true_label"], a["ord_prob"]),
            "auc_b": roc_auc_score(b["true_label"], b["ord_prob"]),
            "auc_diff": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "p_value_two_sided": p,
            "test_type": "independent bootstrap across non-overlapping admission strata",
        })

race_rows = []
for task in TASKS:
    task_df = df[df["task"] == task].dropna(subset=["true_label", "ord_prob"])
    white = task_df[task_df["race_group"] == "White"]
    for group in ["Asian", "Black", "Hispanic", "Other", "Unknown"]:
        other = task_df[task_df["race_group"] == group]
        if len(other) < 50 or other["true_label"].sum() < 5:
            continue
        diff, lo, hi, p = independent_bootstrap_auc_diff(
            other["true_label"].values, other["ord_prob"].values,
            white["true_label"].values, white["ord_prob"].values,
        )
        race_rows.append({
            "dataset": "MIMIC-IV test",
            "task": task,
            "model": "ordering_only",
            "comparison": f"{group} - White",
            "group_a": group,
            "group_b": "White",
            "n_a": int(len(other)),
            "n_pos_a": int(other["true_label"].sum()),
            "n_b": int(len(white)),
            "n_pos_b": int(white["true_label"].sum()),
            "auc_a": roc_auc_score(other["true_label"], other["ord_prob"]),
            "auc_b": roc_auc_score(white["true_label"], white["ord_prob"]),
            "auc_diff": diff,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "p_value_two_sided": p,
            "test_type": "independent bootstrap across non-overlapping race strata",
        })

pd.DataFrame(model_rows).to_csv(OUT / "mimic_paired_model_auc_comparisons.csv", index=False)
pd.DataFrame(timing_rows).to_csv(OUT / "mimic_timing_auc_comparison_tests.csv", index=False)
pd.DataFrame(race_rows).to_csv(OUT / "mimic_race_auc_comparison_tests.csv", index=False)

print(OUT / "mimic_paired_model_auc_comparisons.csv")
print(OUT / "mimic_timing_auc_comparison_tests.csv")
print(OUT / "mimic_race_auc_comparison_tests.csv")
