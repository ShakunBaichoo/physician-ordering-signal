# physician-ordering-signal

Code accompanying the manuscript:

> **Laboratory test-ordering patterns carry an early prognostic signal in
> electronic health records**
>
> Baichoo *et al.*

This repository contains the experiment scripts used to train, evaluate, and
audit the LightGBM and GRU models reported in the paper. It does **not**
distribute any patient data — all data must be obtained directly from the
original custodians (see *Data access* below).

## Repository layout

```
physician-ordering-signal/
├── experiments/        # all analysis scripts (A_*.py … X_*.py)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

Every script in `experiments/` expects to be run from a parent project
directory that contains a `data/processed/` folder and writes results to
`1_ordering_paper/results/`. Specifically, each script computes:

```python
ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "1_ordering_paper" / "results" / "<experiment>"
```

To reproduce, clone into a project root with the following layout:

```
<project_root>/
├── data/
│   └── processed/
│       ├── cohort.parquet
│       ├── labels.parquet
│       ├── static.parquet
│       └── timeseries/
│           ├── labs.parquet
│           └── vitals.parquet
└── 1_ordering_paper/
    └── experiments/    # this repository, cloned here
```

## Data access

| Cohort  | Source                                                                                  | Access                           |
|---------|-----------------------------------------------------------------------------------------|----------------------------------|
| MIMIC-IV | PhysioNet — <https://physionet.org/content/mimiciv/>                                    | Credentialed access required     |
| eICU-CRD | PhysioNet — <https://physionet.org/content/eicu-crd/>                                   | Credentialed access required     |
| MC-MED   | Stanford AIMI — <https://aimi.stanford.edu/datasets/mc-med-multimodal-clinical-monitoring-emergency-department> | Data-use agreement required      |

Feature extraction (4-hour binned labs/vitals with measurement-count `_obs`
columns) is described in the manuscript Methods.

## Environment

Python 3.10+ recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

All random seeds are set to `42` inside each script.

## Reproducing the manuscript

Scripts are named alphabetically by the experiment they implement (see the
manuscript Methods and Supplementary Methods for descriptions). The main
dependencies between scripts are: `A_ordering_signal` produces an
ordering-feature CSV consumed by several later scripts, and
`H_cci_stratified` produces test-set predictions consumed by `I`, `J`, `M`,
and `R`. Within these constraints, a reasonable execution order is:

```bash
cd experiments

# 1. Main comparison and ordering-feature derivation
python A_ordering_signal.py
python A_model_sensitivity.py
python F_paper_improvements.py
python F_parts345.py

# 2. Stratified models + downstream consumers
python H_cci_stratified.py
python I_equity_analysis.py
python J_timing_effects.py
python M_decision_curve.py
python R_calibration.py

# 3. Equity, attribution and mechanism
python B_conformal_equity.py
python D_shapley_ordering.py
python E_modality_shapley.py
python K_mechanistic.py
python O_behavioral_fingerprint.py
python O_behavioral_fingerprint_figure.py

# 4. Phenotype / case-control / timing
python G_case_control_matching.py
python L_triage_window.py
python S_temporal_trajectory.py
python T_dose_response.py

# 5. Validation (temporal, external, ED)
python N_temporal_validation.py
python V_external_validation.py
python Q_mcmed_validation.py
python P_ed_triage_validation.py
python P_ed_triage_figure.py

# 6. Causal / instrumental-variable analysis
python U_hospital_iv.py

# 7. Sensitivity analyses
python W_leakage_controlled.py
python X_temporal_gru.py
python generate_available_paired_stats.py
```

Each script writes `metrics.json`, `*.csv` and/or `*.parquet` artefacts
into `1_ordering_paper/results/<experiment>/` (as defined by the `OUT`
path inside each script).

## Citation

If you use this code, please cite the manuscript (citation block in
`CITATION.cff`).

## License

MIT — see `LICENSE`.
