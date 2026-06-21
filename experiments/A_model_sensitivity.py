#!/usr/bin/env python
"""
Experiment A sensitivity — non-LightGBM model-family check
==========================================================

Purpose:
  Supplementary robustness check for the core Experiment A claim.
  This is NOT an algorithm leaderboard. It holds the same cohort, outcomes,
  patient-level split, and feature-set comparison fixed, then asks whether
  ordering-only signal is still present using simple non-LightGBM classifiers.

Models:
  1. L2-regularised logistic regression (linear baseline)
  2. Constrained random forest (non-boosted tree ensemble)

Feature sets:
  - values_only: static + lab/vital value summaries
  - ordering_only: static + lab/vital observation-count summaries
  - combined: static + values + ordering

Outputs:
  results/A_model_sensitivity/model_sensitivity_metrics.csv
  results/A_model_sensitivity/model_sensitivity_metrics.json
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "1_ordering_paper" / "results" / "A_model_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]
SEED = 42


def aggregate_values(df_long: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Mean over 12 time bins for all non-observation features."""
    val_cols = [
        c for c in df_long.columns
        if c not in ("hadm_id", "time_bin") and not c.endswith("_obs")
    ]
    out = df_long.groupby("hadm_id", sort=False)[val_cols].mean().reset_index()
    out.columns = ["hadm_id"] + [f"{c}__{suffix}" for c in val_cols]
    return out


def aggregate_ordering(df_long: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Compact ordering representation for model-family sensitivity:
      - total observed count per test over 48h
      - fraction of 4h bins with >=1 measurement per test
      - admission-level ordering breadth/intensity

    The primary LightGBM experiment uses a richer ordering representation,
    including per-test temporal slopes. This compact version is deliberate:
    it makes the sensitivity analysis faster and easier to interpret.
    """
    obs_cols = [c for c in df_long.columns if c.endswith("_obs")]
    sums = df_long.groupby("hadm_id", sort=False)[obs_cols].sum()
    intensity = df_long.assign(**{c: (df_long[c] > 0).astype(float) for c in obs_cols})
    intensity = intensity.groupby("hadm_id", sort=False)[obs_cols].mean()

    sums.columns = [f"total_{c}__{suffix}" for c in obs_cols]
    intensity.columns = [f"intensity_{c}__{suffix}" for c in obs_cols]

    binary = df_long[obs_cols].gt(0).astype(float)
    summary = pd.DataFrame({
        "hadm_id": df_long["hadm_id"].values,
        f"tests_per_bin__{suffix}": binary.sum(axis=1).values,
        f"any_test_bin__{suffix}": binary.any(axis=1).astype(float).values,
    })
    summary = summary.groupby("hadm_id", sort=False).agg({
        f"tests_per_bin__{suffix}": "mean",
        f"any_test_bin__{suffix}": "mean",
    })

    out = sums.join(intensity).join(summary).reset_index()
    return out


def patient_split(df: pd.DataFrame):
    pat = df.groupby("subject_id")["mortality"].max().reset_index()
    pat = pat.sample(frac=1, random_state=SEED)
    n = len(pat)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    train_s = set(pat.iloc[:n_train]["subject_id"])
    val_s = set(pat.iloc[n_train:n_train + n_val]["subject_id"])
    test_s = set(pat.iloc[n_train + n_val:]["subject_id"])
    return (
        df["subject_id"].isin(train_s),
        df["subject_id"].isin(val_s),
        df["subject_id"].isin(test_s),
    )


def score_model(model, X, y) -> dict[str, float]:
    p = model.predict_proba(X)[:, 1]
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
    }


def fit_eval(model_key: str, X_tr, y_tr, X_va, y_va, X_te, y_te):
    if model_key == "logistic_l2":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="saga",
                max_iter=1000,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
        )
    elif model_key == "random_forest":
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=14,
                min_samples_leaf=50,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=SEED,
                n_jobs=-1,
            ),
        )
    else:
        raise ValueError(model_key)

    model.fit(X_tr, y_tr)
    return model, score_model(model, X_va, y_va), score_model(model, X_te, y_te)


def main():
    print("=" * 70)
    print("Experiment A sensitivity — logistic regression and random forest")
    print("=" * 70)

    print("\n[1] Loading processed data")
    cohort = pd.read_parquet(DATA / "cohort.parquet")[["hadm_id", "subject_id"]]
    labels = pd.read_parquet(DATA / "labels.parquet")
    static = pd.read_parquet(DATA / "static.parquet")
    labs = pd.read_parquet(DATA / "timeseries" / "labs.parquet")
    vitals = pd.read_parquet(DATA / "timeseries" / "vitals.parquet")

    print("[2] Aggregating compact feature sets")
    labs_val = aggregate_values(labs, "lab")
    vitals_val = aggregate_values(vitals, "vit")
    labs_ord = aggregate_ordering(labs, "lab")
    vitals_ord = aggregate_ordering(vitals, "vit")

    base = cohort.merge(labels, on=["hadm_id", "subject_id"]).merge(
        static, on="hadm_id", how="left"
    )
    df_val = base.merge(labs_val, on="hadm_id", how="left").merge(
        vitals_val, on="hadm_id", how="left"
    )
    df_ord = base.merge(labs_ord, on="hadm_id", how="left").merge(
        vitals_ord, on="hadm_id", how="left"
    )
    df_both = df_val.merge(labs_ord, on="hadm_id", how="left").merge(
        vitals_ord, on="hadm_id", how="left"
    )

    meta_cols = ["hadm_id", "subject_id"] + TASKS
    datasets = {
        "values_only": df_val,
        "ordering_only": df_ord,
        "combined": df_both,
    }

    tr_m, va_m, te_m = patient_split(df_both)
    train_subjects = set(df_both.loc[tr_m, "subject_id"])
    val_subjects = set(df_both.loc[va_m, "subject_id"])
    test_subjects = set(df_both.loc[te_m, "subject_id"])
    print(f"[3] Split: train={len(train_subjects):,} patients, "
          f"val={len(val_subjects):,}, test={len(test_subjects):,}")
    print(f"    Rows: train={tr_m.sum():,}, val={va_m.sum():,}, test={te_m.sum():,}")

    rows = []
    for feature_set, df in datasets.items():
        feat_cols = [c for c in df.columns if c not in meta_cols]
        tr = df["subject_id"].isin(train_subjects)
        va = df["subject_id"].isin(val_subjects)
        te = df["subject_id"].isin(test_subjects)
        print(f"\n[4] Feature set: {feature_set} ({len(feat_cols)} features)")

        for task in TASKS:
            ok_tr = tr & df[task].notna()
            ok_va = va & df[task].notna()
            ok_te = te & df[task].notna()
            X_tr = df.loc[ok_tr, feat_cols].astype("float32")
            X_va = df.loc[ok_va, feat_cols].astype("float32")
            X_te = df.loc[ok_te, feat_cols].astype("float32")
            y_tr = df.loc[ok_tr, task].astype(int).values
            y_va = df.loc[ok_va, task].astype(int).values
            y_te = df.loc[ok_te, task].astype(int).values

            for model_key in ["logistic_l2", "random_forest"]:
                print(f"    {task:<16} {model_key:<14}", end="", flush=True)
                _, val_s, test_s = fit_eval(model_key, X_tr, y_tr, X_va, y_va, X_te, y_te)
                row = {
                    "model": model_key,
                    "feature_set": feature_set,
                    "task": task,
                    "n_train": int(len(y_tr)),
                    "n_val": int(len(y_va)),
                    "n_test": int(len(y_te)),
                    "n_features": int(len(feat_cols)),
                    "val_auroc": round(val_s["auroc"], 4),
                    "val_auprc": round(val_s["auprc"], 4),
                    "test_auroc": round(test_s["auroc"], 4),
                    "test_auprc": round(test_s["auprc"], 4),
                }
                rows.append(row)
                print(f" test AUROC={row['test_auroc']:.4f} AUPRC={row['test_auprc']:.4f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model_sensitivity_metrics.csv", index=False)
    with open(OUT / "model_sensitivity_metrics.json", "w") as f:
        json.dump({"results": rows}, f, indent=2)

    pivot = res.pivot_table(
        index=["model", "task"],
        columns="feature_set",
        values="test_auroc",
        aggfunc="first",
    ).reset_index()
    pivot.to_csv(OUT / "model_sensitivity_auroc_pivot.csv", index=False)

    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
