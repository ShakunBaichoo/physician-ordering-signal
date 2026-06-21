#!/usr/bin/env python
"""
Experiment W -- Leakage-controlled re-analysis for AKI and sepsis
================================================================
Sensitivity analysis for potential circular outcome ascertainment.

Concern
-------
* AKI is defined (partly) by KDIGO creatinine rise >= 0.3 mg/dL within 48h.
  The `creatinine` measurement-count channel is therefore an outcome-defining
  test that also enters the ORDERING feature set -> potential circularity.
* Sepsis is defined by ICD-10 A41.x + blood culture + IV antibiotics within 72h.
  - Blood culture is microbiology (NOT one of the 35 lab channels) -> it is
    absent from the ordering features. We verify and state this.
  - The actual label-overlapping feature is the STATIC `med_antibiotic`
    (and antifungal/antiviral) flag, built from prescriptions within
    admittime+48h (subset of the 72h sepsis window). This static flag is shared
    by ALL models (values / ordering / combined), so it is a leak in the
    baseline itself, not specific to ordering.

Approach
--------
Exclude the outcome-defining tests/flags from the features and re-train, then
report the change in AUROC/AUPRC versus the full-feature baseline. Same
subject-level split, same LightGBM config, same seed -> a clean paired delta.

Configurations
--------------
AKI  : ordering features exclude {creatinine}; sensitivity excludes {creatinine,bun}.
Sepsis: static features exclude {med_antibiotic, med_antifungal, med_antiviral};
        additional variant also drops ordering channels {lactate, wbc, bands}.

For each (task, model in {values_only, ordering_only, combined}) we train a
FULL model and a LEAKAGE-CONTROLLED model on the identical split and compute a
two-sided paired bootstrap test on the AUROC difference (1000 resamples).

Outputs (results/W_leakage_controlled/):
  - leakage_controlled_metrics.json
  - leakage_controlled_summary.csv

NB: This mirrors experiments/A_ordering_signal.py feature definitions exactly
(verified by reproducing the A baseline AUROCs), but uses a vectorised feature
builder (every admission has exactly 12 4-hour bins) for speed.
"""

import json
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
TS   = DATA / "timeseries"
OUT  = ROOT / "1_ordering_paper" / "results" / "W_leakage_controlled"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
NBINS = 12
N_BOOT = 1000

# Outcome-defining tests / flags to exclude (matches Methods outcome definitions)
AKI_ORDER_EXCLUDE          = ["creatinine"]            # KDIGO creatinine rise
AKI_ORDER_EXCLUDE_SENS     = ["creatinine", "bun"]     # + correlated renal panel
SEPSIS_STATIC_EXCLUDE      = ["med_antibiotic", "med_antifungal", "med_antiviral"]
SEPSIS_ORDER_EXCLUDE_EXTRA = ["lactate", "wbc", "bands"]  # canonical sepsis workup labs


# ── 1. Load obs/value tensors (vectorised, fixed 12 bins) ─────────────────────
def load_long(name):
    """Return (hadm_ids sorted, obs_tensor (N,12,C_obs), obs_channels,
                value_matrix (N, V) = mean across bins, value_names)."""
    con = duckdb.connect()
    cols = con.execute(
        f"SELECT * FROM '{TS / (name + '.parquet')}' LIMIT 0"
    ).fetchdf().columns.tolist()
    con.close()
    obs_cols = [c for c in cols if c.endswith("_obs")]
    val_cols = [c for c in cols if c not in ("hadm_id", "time_bin")
                and not c.endswith("_obs")]
    sel = ["hadm_id", "time_bin"] + obs_cols + val_cols
    df = pd.read_parquet(TS / f"{name}.parquet", columns=sel)
    df = df.sort_values(["hadm_id", "time_bin"], kind="stable")
    assert len(df) % NBINS == 0, f"{name}: row count not divisible by {NBINS} bins"
    hh = df["hadm_id"].values.reshape(-1, NBINS)
    assert (hh == hh[:, :1]).all(), f"{name}: non-constant hadm within a 12-bin block"
    assert df["time_bin"].values.reshape(-1, NBINS)[0].tolist() == sorted(
        df["time_bin"].values.reshape(-1, NBINS)[0].tolist()), f"{name}: bins not sorted"
    hadm = hh[:, 0]
    N = hadm.shape[0]
    obs = df[obs_cols].values.reshape(N, NBINS, len(obs_cols)).astype("float32")
    valmat = df[val_cols].values.reshape(N, NBINS, len(val_cols)).astype("float32")
    valmean = np.nanmean(valmat, axis=1)  # mean across bins (matches A groupby.mean)
    obs_channels = [c[:-4] for c in obs_cols]
    return hadm, obs, obs_channels, valmean, [f"{c}__{name}" for c in val_cols]


def ordering_features(obs, channels, name, exclude=()):
    """Vectorised reproduction of A_ordering_signal.extract_ordering_features.
    Per-channel: total / intensity / slope. Admission-level summaries are
    recomputed over the INCLUDED channels only (strict leakage control)."""
    keep = [i for i, ch in enumerate(channels) if ch not in set(exclude)]
    o = obs[:, :, keep]                              # (N,12,C')
    t = np.arange(NBINS, dtype="float32")
    tc = t - t.mean()
    denom = float((tc ** 2).sum())

    total = o.sum(1)                                 # (N,C')
    intensity = (o > 0).mean(1)                      # (N,C')
    slope = ((o - o.mean(1, keepdims=True)) * tc[None, :, None]).sum(1) / denom

    binary = (o > 0).astype("float32")
    ord_intensity = binary.mean((1, 2))              # (N,)
    ord_diversity = (o.sum(1) > 0).sum(1)            # (N,)
    tests_per_bin = binary.sum(2)                    # (N,12)
    ord_breadth = tests_per_bin.mean(1)
    ord_escalation = ((tests_per_bin - tests_per_bin.mean(1, keepdims=True))
                      * tc[None, :]).sum(1) / denom

    kept = [channels[i] for i in keep]
    feats = {}
    for j, ch in enumerate(kept):
        feats[f"total_obs_{ch}__{name}"] = total[:, j]
        feats[f"intensity_{ch}__{name}"] = intensity[:, j]
        feats[f"slope_{ch}__{name}"] = slope[:, j]
    feats[f"ordering_intensity__{name}"] = ord_intensity
    feats[f"ordering_diversity__{name}"] = ord_diversity.astype("float32")
    feats[f"ordering_breadth__{name}"] = ord_breadth
    feats[f"ordering_escalation__{name}"] = ord_escalation
    return pd.DataFrame(feats)


print("Loading time-series tensors...")
h_lab, obs_lab, ch_lab, val_lab, valnames_lab = load_long("labs")
h_vit, obs_vit, ch_vit, val_vit, valnames_vit = load_long("vitals")
assert np.array_equal(h_lab, h_vit), "lab/vital hadm ordering mismatch"
hadm = h_lab
print(f"  N admissions: {len(hadm):,} | lab channels {len(ch_lab)} | vital channels {len(ch_vit)}")
print(f"  creatinine in lab channels: {'creatinine' in ch_lab} | "
      f"lactate/wbc/bands: {[c in ch_lab for c in ['lactate','wbc','bands']]}")

# Value-feature frame (mean across bins) -- identical across all configs
val_df = pd.DataFrame(np.hstack([val_lab, val_vit]),
                      columns=valnames_lab + valnames_vit)
val_df.insert(0, "hadm_id", hadm)

# Labels / static -- labels.parquet already carries subject_id + the 4 tasks
base = pd.read_parquet(DATA / "labels.parquet")   # hadm_id, subject_id, 4 outcomes
static = pd.read_parquet(DATA / "static.parquet")
TASKS_ALL = ["mortality", "readmission_30d", "aki", "sepsis"]


# ── 2. Subject-level split (identical to A_ordering_signal) ────────────────────
order = pd.DataFrame({"hadm_id": hadm}).merge(base, on="hadm_id", how="left")
pat = order.groupby("subject_id")["mortality"].max().reset_index()
pat = pat.sample(frac=1, random_state=SEED)
n = len(pat)
n_tr, n_va = int(0.70 * n), int(0.15 * n)
train_s = set(pat.iloc[:n_tr]["subject_id"])
val_s   = set(pat.iloc[n_tr:n_tr + n_va]["subject_id"])
test_s  = set(pat.iloc[n_tr + n_va:]["subject_id"])
assert train_s.isdisjoint(val_s) and train_s.isdisjoint(test_s) and val_s.isdisjoint(test_s), \
    "subject overlap across splits"
subj = order["subject_id"].values
tr_mask = np.array([s in train_s for s in subj])
va_mask = np.array([s in val_s for s in subj])
te_mask = np.array([s in test_s for s in subj])
assert (tr_mask & va_mask).sum() == 0 and (tr_mask & te_mask).sum() == 0 and (va_mask & te_mask).sum() == 0
assert tr_mask.sum() + va_mask.sum() + te_mask.sum() == len(subj), "split does not partition all rows"
print(f"Split -- Train {tr_mask.sum():,} | Val {va_mask.sum():,} | Test {te_mask.sum():,}")


# ── 3. Train / eval helpers ───────────────────────────────────────────────────
def train_eval(X, y, task):
    Xtr, Xva, Xte = X[tr_mask], X[va_mask], X[te_mask]
    ytr, yva, yte = y[tr_mask], y[va_mask], y[te_mask]
    ok_tr, ok_va, ok_te = ~np.isnan(ytr), ~np.isnan(yva), ~np.isnan(yte)
    prev = ytr[ok_tr].mean()
    pos_w = (1 - prev) / prev
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=127,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, scale_pos_weight=pos_w,
        n_jobs=-1, random_state=SEED, verbose=-1, metric="auc")
    model.fit(Xtr[ok_tr], ytr[ok_tr],
              eval_set=[(Xva[ok_va], yva[ok_va])],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    prob = model.predict_proba(Xte[ok_te])[:, 1]
    return {
        "auroc": float(roc_auc_score(yte[ok_te], prob)),
        "auprc": float(average_precision_score(yte[ok_te], prob)),
        "prob": prob, "y": yte[ok_te].astype(int),
    }


def paired_boot_delta(y, p_full, p_ctrl, n_boot=N_BOOT, seed=SEED):
    """Two-sided paired bootstrap on AUROC(ctrl) - AUROC(full).
    Both models scored on the SAME resampled indices (truly paired); one-class
    resamples are skipped (not scored as 0)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():           # skip degenerate one-class resample
            continue
        diffs.append(roc_auc_score(yb, p_ctrl[idx]) - roc_auc_score(yb, p_full[idx]))
    diffs = np.asarray(diffs)
    obs = roc_auc_score(y, p_ctrl) - roc_auc_score(y, p_full)
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), min(p, 1.0)


def build_X(model_kind, ord_exclude=(), static_exclude=()):
    """Assemble feature matrix for a given config. Returns (X ndarray, names)."""
    parts = [static.set_index("hadm_id").drop(columns=list(static_exclude),
                                              errors="ignore")]
    if model_kind in ("values_only", "combined"):
        parts.append(val_df.set_index("hadm_id"))
    if model_kind in ("ordering_only", "combined"):
        lab_o = ordering_features(obs_lab, ch_lab, "lab", exclude=ord_exclude)
        vit_o = ordering_features(obs_vit, ch_vit, "vit", exclude=ord_exclude)
        ord_df = pd.concat([lab_o, vit_o], axis=1)
        ord_df.index = hadm
        parts.append(ord_df)
    # align all on hadm order
    idx = pd.Index(hadm, name="hadm_id")
    mat = pd.concat([p.reindex(idx) for p in parts], axis=1)
    cols = mat.columns.tolist()
    # hard-verify the exclusions actually took effect
    if model_kind in ("ordering_only", "combined"):
        for ch in ord_exclude:
            assert not any(f"_{ch}__" in c for c in cols), \
                f"excluded ordering channel '{ch}' still present in {model_kind}"
    for sc in static_exclude:
        assert sc not in cols, f"excluded static col '{sc}' still present in {model_kind}"
    assert mat.index.equals(idx), "feature matrix row order != hadm order (label misalignment risk)"
    return mat.values.astype("float32"), cols


# ── 4. Run configurations ─────────────────────────────────────────────────────
CONFIGS = {
    "aki": {
        "task": "aki",
        "full": dict(ord_exclude=(), static_exclude=()),
        "controlled": dict(ord_exclude=AKI_ORDER_EXCLUDE, static_exclude=()),
        "sensitivity": dict(ord_exclude=AKI_ORDER_EXCLUDE_SENS, static_exclude=()),
        "note": "ordering features exclude creatinine (KDIGO label-defining test)",
    },
    "sepsis": {
        "task": "sepsis",
        "full": dict(ord_exclude=(), static_exclude=()),
        "controlled": dict(ord_exclude=(), static_exclude=SEPSIS_STATIC_EXCLUDE),
        "sensitivity": dict(ord_exclude=SEPSIS_ORDER_EXCLUDE_EXTRA,
                            static_exclude=SEPSIS_STATIC_EXCLUDE),
        "note": ("static med_antibiotic/antifungal/antiviral overlap the 72h sepsis "
                 "label; blood culture is microbiology and absent from ordering features"),
    },
}
MODELS = ["values_only", "ordering_only", "combined"]

results = {}
rows = []
for cfg_name, cfg in CONFIGS.items():
    task = cfg["task"]
    y = base.set_index("hadm_id").reindex(hadm)[task].values.astype(float)
    results[cfg_name] = {"note": cfg["note"], "models": {}}
    print(f"\n{'='*70}\n  {cfg_name.upper()}  --  {cfg['note']}\n{'='*70}")
    for model_kind in MODELS:
        results[cfg_name]["models"][model_kind] = {}
        # full
        Xf, namesf = build_X(model_kind, **cfg["full"])
        rf = train_eval(Xf, y, task)
        # controlled
        Xc, namesc = build_X(model_kind, **cfg["controlled"])
        rc = train_eval(Xc, y, task)
        obs_d, lo, hi, pval = paired_boot_delta(rf["y"], rf["prob"], rc["prob"])
        # sensitivity
        Xs, namess = build_X(model_kind, **cfg["sensitivity"])
        rs = train_eval(Xs, y, task)
        obs_ds, los, his, pvals = paired_boot_delta(rf["y"], rf["prob"], rs["prob"])

        results[cfg_name]["models"][model_kind] = {
            "n_features_full": len(namesf),
            "n_features_controlled": len(namesc),
            "full":        {"auroc": round(rf["auroc"], 4), "auprc": round(rf["auprc"], 4)},
            "controlled":  {"auroc": round(rc["auroc"], 4), "auprc": round(rc["auprc"], 4)},
            "sensitivity": {"auroc": round(rs["auroc"], 4), "auprc": round(rs["auprc"], 4)},
            "delta_auroc_controlled": round(obs_d, 4),
            "delta_ci_controlled": [round(lo, 4), round(hi, 4)],
            "p_controlled": round(pval, 4),
            "delta_auroc_sensitivity": round(obs_ds, 4),
            "p_sensitivity": round(pvals, 4),
        }
        m = results[cfg_name]["models"][model_kind]
        print(f"  {model_kind:14} full {rf['auroc']:.4f} -> ctrl {rc['auroc']:.4f} "
              f"(d={obs_d:+.4f} [{lo:+.4f},{hi:+.4f}] p={pval:.3f}) "
              f"| sens {rs['auroc']:.4f} (d={obs_ds:+.4f})")
        rows.append({
            "task": task, "model": model_kind,
            "auroc_full": round(rf["auroc"], 4), "auroc_controlled": round(rc["auroc"], 4),
            "delta_auroc": round(obs_d, 4),
            "delta_ci_low": round(lo, 4), "delta_ci_high": round(hi, 4),
            "p_value": round(pval, 4),
            "auprc_full": round(rf["auprc"], 4), "auprc_controlled": round(rc["auprc"], 4),
            "auroc_sensitivity": round(rs["auroc"], 4), "delta_sensitivity": round(obs_ds, 4),
        })

with open(OUT / "leakage_controlled_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame(rows).to_csv(OUT / "leakage_controlled_summary.csv", index=False)
print(f"\nSaved -> {OUT}/leakage_controlled_metrics.json + leakage_controlled_summary.csv")
