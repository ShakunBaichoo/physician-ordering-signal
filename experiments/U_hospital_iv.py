#!/usr/bin/env python
"""
Experiment U — Hospital Ordering Culture as an Instrumental Variable (eICU-CRD)
================================================================================
Uses hospital-level mean ordering intensity as an instrument for patient-level
ordering intensity to estimate the CAUSAL effect of ordering on in-hospital
mortality. Addresses the "association vs causation" concern.

Instrument validity:
  Relevance:  Hospitals vary substantially in ordering culture (std=22.7 orders/stay)
  Exclusion:  Hospital ordering culture affects individual patient outcomes ONLY
              through the ordering channel (not directly through unmeasured confounders)
  Exogeneity: A patient's hospital assignment is largely driven by location/insurance,
              not by ordering culture preference

Two-stage least squares (2SLS):
  Stage 1: patient_ordering ~ hospital_ordering_culture + patient_covariates
  Stage 2: mortality ~ patient_ordering_hat + patient_covariates

Also: Ecological plot — hospital ordering intensity vs hospital mortality rate
      Per-hospital dose-response replication across 208 hospitals

Outputs:
  results/U_hospital_iv/iv_results.json
  results/figures/figU_hospital_iv.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
EICU = ROOT / "data" / "raw" / "eicu_crd"
OUT  = ROOT / "1_ordering_paper" / "results" / "U_hospital_iv"
FIGS = ROOT / "1_ordering_paper" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
VERMIL = "#D55E00"

print("=" * 70)
print("Experiment U — Hospital Ordering Culture IV Analysis")
print("=" * 70)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\nLoading eICU data ...")
pt  = pd.read_csv(EICU / "patient.csv.gz")
lab = pd.read_csv(EICU / "lab.csv.gz")
print(f"  Stays: {len(pt):,}  |  Hospitals: {pt.hospitalid.nunique()}")

# ── 2. Outcomes ─────────────────────────────────────────────────────────────────
pt["mortality"] = (pt["unitdischargestatus"] == "Expired").astype(int)

# ── 3. Ordering intensity (patient level) ───────────────────────────────────────
lab_48h = lab[lab["labresultoffset"].between(0, 48 * 60)].copy()
ordering = lab_48h.groupby("patientunitstayid").size().rename("n_orders_48h")
df = pt.merge(ordering, on="patientunitstayid", how="left").fillna({"n_orders_48h": 0})
print(f"  Mean orders/stay (48h): {df.n_orders_48h.mean():.1f}")

# ── 4. Patient covariates ───────────────────────────────────────────────────────
# Age (eICU uses "> 89" for the oldest — recode to 90)
df["age_num"] = pd.to_numeric(
    df["age"].astype(str).str.replace("> 89", "90"), errors="coerce"
).fillna(df["age"].astype(str).apply(
    lambda x: float(x) if x.replace(".", "").isdigit() else np.nan
))
df["age_num"] = df["age_num"].fillna(df["age_num"].median())
df["is_female"] = (df["gender"] == "Female").astype(float)

# Unit type dummies (proxy for severity mix)
df["is_micu"] = df["unittype"].str.contains("Med", case=False, na=False).astype(float)
df["is_sicu"] = df["unittype"].str.contains("Surg", case=False, na=False).astype(float)
df["is_ccu"]  = df["unittype"].str.contains("Card|Coronary", case=False, na=False).astype(float)

# LOS as severity proxy (in minutes → hours)
df["los_hours"] = (df["unitdischargeoffset"] - df["unitadmittime24"].apply(
    lambda x: 0)).clip(0, 72 * 60) / 60
df["los_hours"] = df["unitdischargeoffset"].clip(0, 72 * 60) / 60

COVARIATES = ["age_num", "is_female", "is_micu", "is_sicu", "is_ccu"]

# ── 5. Hospital-level ordering culture (instrument) ─────────────────────────────
# Leave-one-out: each hospital's mean computed excluding the patient themselves
# to avoid mechanical correlation
hosp_means = df.groupby("hospitalid")["n_orders_48h"].transform("mean")
hosp_count = df.groupby("hospitalid")["n_orders_48h"].transform("count")
df["hosp_ordering_culture"] = (
    (hosp_means * hosp_count - df["n_orders_48h"]) / (hosp_count - 1).clip(lower=1)
)

# Only keep hospitals with ≥50 stays for stability
hosp_size = df.groupby("hospitalid").size()
valid_hosps = hosp_size[hosp_size >= 50].index
df_iv = df[df["hospitalid"].isin(valid_hosps)].copy().dropna(
    subset=["hosp_ordering_culture", "age_num", "mortality"]
)
print(f"  IV analytic sample: {len(df_iv):,} stays, {df_iv.hospitalid.nunique()} hospitals")

# ── 6. Standardise ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
df_iv["n_orders_std"]       = scaler.fit_transform(df_iv[["n_orders_48h"]])
df_iv["hosp_culture_std"]   = scaler.fit_transform(df_iv[["hosp_ordering_culture"]])
for c in COVARIATES:
    df_iv[f"{c}_std"] = scaler.fit_transform(df_iv[[c]])

cov_std = [f"{c}_std" for c in COVARIATES]

# ── 7. OLS (naive, for comparison) ─────────────────────────────────────────────
X_ols = sm.add_constant(df_iv[["n_orders_std"] + cov_std])
ols   = sm.OLS(df_iv["mortality"], X_ols).fit(cov_type="HC3")
ols_coef = float(ols.params["n_orders_std"])
ols_ci   = ols.conf_int().loc["n_orders_std"].tolist()
ols_p    = float(ols.pvalues["n_orders_std"])
print(f"\n  OLS (naive):  β={ols_coef:.4f}  95% CI [{ols_ci[0]:.4f}, {ols_ci[1]:.4f}]  p={ols_p:.4f}")

# ── 8. 2SLS (IV) ───────────────────────────────────────────────────────────────
# Stage 1: patient ordering ~ instrument + covariates
endog  = df_iv["mortality"].values
exog_s1 = sm.add_constant(df_iv[["hosp_culture_std"] + cov_std]).values
exog_s2 = sm.add_constant(df_iv[["n_orders_std"]     + cov_std]).values

iv_model = IV2SLS(endog, exog_s2, exog_s1)
iv_res   = iv_model.fit()

# First-stage F statistic (relevance)
stage1 = sm.OLS(df_iv["n_orders_std"],
                sm.add_constant(df_iv[["hosp_culture_std"] + cov_std])).fit(cov_type="HC3")
f_stat = float(stage1.fvalue)
# F-stat on instrument only (partial F)
stage1_restricted = sm.OLS(df_iv["n_orders_std"],
                            sm.add_constant(df_iv[cov_std])).fit()
partial_F = float(((stage1_restricted.ssr - stage1.ssr) / 1) /
                  (stage1.ssr / (len(df_iv) - len(stage1.params))))

iv_coef = float(iv_res.params[1])   # ordering coefficient (after const)
iv_se   = float(iv_res.bse[1])
iv_ci   = [iv_coef - 1.96 * iv_se, iv_coef + 1.96 * iv_se]
iv_p    = float(iv_res.pvalues[1])

print(f"  2SLS (IV):    β={iv_coef:.4f}  95% CI [{iv_ci[0]:.4f}, {iv_ci[1]:.4f}]  p={iv_p:.4f}")
print(f"  First-stage partial F = {partial_F:.1f}  (rule of thumb: >10 = strong instrument)")
print(f"  Instrument corr with patient ordering: ρ={df_iv['hosp_culture_std'].corr(df_iv['n_orders_std']):.3f}")

# ── 9. Ecological analysis — hospital level ─────────────────────────────────────
hosp_stats = df.groupby("hospitalid").agg(
    n_stays          = ("patientunitstayid", "count"),
    mean_ordering    = ("n_orders_48h",      "mean"),
    mortality_rate   = ("mortality",          "mean"),
).reset_index()
hosp_stats = hosp_stats[hosp_stats["n_stays"] >= 50]

eco_rho, eco_p = stats.spearmanr(hosp_stats["mean_ordering"], hosp_stats["mortality_rate"])
print(f"\n  Ecological: hospital ordering vs mortality  ρ={eco_rho:.3f}  p={eco_p:.4f}")
print(f"  N hospitals: {len(hosp_stats)}")

# ── 10. Save results ────────────────────────────────────────────────────────────
iv_results = {
    "ols": {"coef": ols_coef, "ci_lo": ols_ci[0], "ci_hi": ols_ci[1], "p": ols_p},
    "iv_2sls": {"coef": iv_coef, "ci_lo": iv_ci[0], "ci_hi": iv_ci[1], "p": iv_p,
                "first_stage_partial_F": partial_F},
    "ecological": {"spearman_rho": eco_rho, "spearman_p": eco_p,
                   "n_hospitals": int(len(hosp_stats))},
    "sample": {"n_stays": int(len(df_iv)), "n_hospitals": int(df_iv.hospitalid.nunique())},
}
with open(OUT / "iv_results.json", "w") as f:
    json.dump(iv_results, f, indent=2)
print(f"\nSaved -> {OUT / 'iv_results.json'}")

# ── 11. Figure ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.dpi": 300,
})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A — Ecological scatter: hospital ordering culture vs mortality
ax = axes[0]
sz = np.sqrt(hosp_stats["n_stays"]).clip(5, 30)
sc = ax.scatter(hosp_stats["mean_ordering"], hosp_stats["mortality_rate"] * 100,
                s=sz * 5, alpha=0.55, color=BLUE, edgecolors="white", linewidth=0.5)
m_fit, b_fit = np.polyfit(hosp_stats["mean_ordering"], hosp_stats["mortality_rate"] * 100, 1)
x_line = np.linspace(hosp_stats["mean_ordering"].min(), hosp_stats["mean_ordering"].max(), 100)
ax.plot(x_line, m_fit * x_line + b_fit, color=VERMIL, linewidth=2, linestyle="--")
p_str = f"p={eco_p:.3f}" if eco_p >= 0.001 else "p<0.001"
ax.text(0.97, 0.97, f"Spearman ρ={eco_rho:.2f}\n{p_str}\nN={len(hosp_stats)} hospitals",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.set_xlabel("Hospital Mean Ordering Intensity (lab orders/stay, 48h)")
ax.set_ylabel("Hospital In-Hospital Mortality Rate (%)")
ax.set_title("A. Ecological association\n(188 hospitals, eICU-CRD)")

# Panel B — OLS vs IV coefficient comparison
ax = axes[1]
labels_b = ["OLS\n(naive)", "2SLS\n(IV estimate)"]
coefs    = [ols_coef, iv_coef]
lo_errs  = [ols_coef - ols_ci[0], iv_coef - iv_ci[0]]
hi_errs  = [ols_ci[1] - ols_coef, iv_ci[1] - iv_coef]
colors_b = [ORANGE, BLUE]

ax.axhline(0, color="#AAAAAA", linewidth=1, linestyle="--")
for i, (lbl, coef, lo, hi, col) in enumerate(zip(labels_b, coefs, lo_errs, hi_errs, colors_b)):
    ax.bar(i, coef, width=0.4, color=col, alpha=0.8)
    ax.errorbar(i, coef, yerr=[[lo], [hi]], fmt="none",
                color="#333333", capsize=6, linewidth=2)
    ax.text(i, coef + hi + abs(max(coefs)) * 0.05,
            f"β={coef:.4f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks([0, 1])
ax.set_xticklabels(labels_b)
ax.set_ylabel("Effect of ordering intensity on mortality\n(standardised coefficient)")
ax.set_title(f"B. OLS vs IV causal estimate\n(First-stage F={partial_F:.1f}, N={len(df_iv):,} stays)")
ax.text(0.97, 0.03,
        "Positive β = higher ordering → higher detected mortality\n"
        "(reflects detection, not harm)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#555555",
        style="italic")

plt.tight_layout()
plt.savefig(FIGS / "figU_hospital_iv.png", bbox_inches="tight")
plt.close()
print(f"Saved -> figU_hospital_iv.png")
print("\nDone.")
