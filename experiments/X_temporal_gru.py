#!/usr/bin/env python
"""
Experiment X -- Temporal deep model (GRU) on raw 4-hour ordering sequences
=========================================================================
Saturation test for the LightGBM ordering model, which uses only three coarse
per-channel summaries (total count, bin fraction, linear slope).
Does a sequence model that sees the *raw* 12-bin x 46-channel ordering trajectory
extract signal BEYOND those summaries -- i.e. non-linear escalation, bursting,
de-escalation that a linear slope cannot capture?

Design
------
* Input  : raw per-bin measurement-event counts, shape (12 bins, 46 channels)
           = 35 lab channels + 11 vital channels. NO measured values, NO summaries.
* Model  : per-channel standardised counts -> GRU -> last hidden -> MLP -> logit.
* Compare: test AUROC/AUPRC vs the LightGBM ordering-only model from experiment A.
           - GRU ~ LightGBM  => ordering signal SATURATES at simple summaries
                                 (volume, not fine temporal pattern, drives it;
                                  consistent with experiment S).
           - GRU  > LightGBM  => raw temporal trajectory carries extra signal.
  Either outcome informs the choice of LightGBM as the primary model.

Identical subject-level split (seed 42) and outcomes as experiment A. Counts are
standardised using TRAIN statistics only (no leakage). Runs on the laptop GPU
(RTX 5060, 8 GB) -- the whole count tensor (~0.7 GB) is preloaded to CUDA.

Outputs (results/X_temporal_gru/):
  - gru_metrics.json   test AUROC/AUPRC + bootstrap CIs, GRU vs LightGBM baseline
"""

import argparse
import json
import warnings
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parents[2]
DATA = ROOT / "data" / "processed"
TS   = DATA / "timeseries"
OUT  = ROOT / "1_ordering_paper" / "results" / "X_temporal_gru"
OUT.mkdir(parents=True, exist_ok=True)
A_METRICS = ROOT / "1_ordering_paper" / "results" / "A_ordering_signal" / "metrics.json"

NBINS = 12
SEED = 42
TASKS = ["mortality", "readmission_30d", "aki", "sepsis"]


def load_obs(name):
    con = duckdb.connect()
    cols = con.execute(f"SELECT * FROM '{TS / (name + '.parquet')}' LIMIT 0").fetchdf().columns.tolist()
    con.close()
    obs_cols = [c for c in cols if c.endswith("_obs")]
    df = pd.read_parquet(TS / f"{name}.parquet", columns=["hadm_id", "time_bin"] + obs_cols)
    df = df.sort_values(["hadm_id", "time_bin"], kind="stable")
    assert len(df) % NBINS == 0, f"{name}: row count not divisible by {NBINS}"
    hh = df["hadm_id"].values.reshape(-1, NBINS)
    assert (hh == hh[:, :1]).all(), f"{name}: non-constant hadm within a 12-bin block"
    hadm = hh[:, 0]
    obs = df[obs_cols].values.reshape(len(hadm), NBINS, len(obs_cols)).astype("float32")
    return hadm, obs, [c[:-4] for c in obs_cols]


def ordering_summary_features(obs, channels, name):
    """LightGBM-style ordering summaries (matches A_ordering_signal / experiment W):
    per-channel total / intensity / slope + 4 admission-level summaries."""
    t = np.arange(NBINS, dtype="float32"); tc = t - t.mean(); denom = float((tc ** 2).sum())
    total = obs.sum(1)
    intensity = (obs > 0).mean(1)
    slope = ((obs - obs.mean(1, keepdims=True)) * tc[None, :, None]).sum(1) / denom
    binary = (obs > 0).astype("float32")
    tests_per_bin = binary.sum(2)
    cols, mats = [], []
    for j, ch in enumerate(channels):
        cols += [f"total_obs_{ch}__{name}", f"intensity_{ch}__{name}", f"slope_{ch}__{name}"]
        mats += [total[:, j:j + 1], intensity[:, j:j + 1], slope[:, j:j + 1]]
    cols += [f"ordering_intensity__{name}", f"ordering_diversity__{name}",
             f"ordering_breadth__{name}", f"ordering_escalation__{name}"]
    mats += [binary.mean((1, 2))[:, None], (obs.sum(1) > 0).sum(1)[:, None].astype("float32"),
             tests_per_bin.mean(1)[:, None],
             (((tests_per_bin - tests_per_bin.mean(1, keepdims=True)) * tc[None, :]).sum(1) / denom)[:, None]]
    return np.hstack(mats).astype("float32"), cols


class GRUClassifier(nn.Module):
    """GRU over raw per-bin counts; optionally concatenate static baseline
    features to the last hidden state (to match LightGBM's information access)."""
    def __init__(self, n_ch, hidden=128, layers=1, dropout=0.2, static_dim=0):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.static_dim = static_dim
        head_in = hidden + static_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in), nn.Dropout(dropout),
            nn.Linear(head_in, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, x, st=None):              # x: (B, 12, n_ch); st: (B, static_dim)
        out, h = self.gru(x)
        z = out[:, -1, :]
        if self.static_dim:
            z = torch.cat([z, st], dim=1)
        return self.head(z).squeeze(-1)


def boot_ci(y, p, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    a = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        a[b] = roc_auc_score(yb, p[idx]) if yb.min() != yb.max() else np.nan
    a = a[~np.isnan(a)]
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def paired_boot(y, p_a, p_b, n_boot=1000, seed=SEED):
    """Two-sided paired bootstrap on AUROC(p_a) - AUROC(p_b), same resample for both."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(yb, p_a[idx]) - roc_auc_score(yb, p_b[idx]))
    diffs = np.asarray(diffs)
    obs = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    p = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), min(p, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--gru_seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = torch.device(args.device)
    print(f"Device: {dev}")

    # ---- data ----
    h_lab, obs_lab, ch_lab = load_obs("labs")
    h_vit, obs_vit, ch_vit = load_obs("vitals")
    assert np.array_equal(h_lab, h_vit)
    hadm = h_lab
    X = np.concatenate([obs_lab, obs_vit], axis=2)         # (N,12,46) raw counts
    n_ch = X.shape[2]
    print(f"N={len(hadm):,} | channels={n_ch} ({len(ch_lab)} lab + {len(ch_vit)} vital) | seq_len={NBINS}")

    labels = pd.read_parquet(DATA / "labels.parquet")      # hadm_id, subject_id, tasks
    lab = pd.DataFrame({"hadm_id": hadm}).merge(labels, on="hadm_id", how="left")

    # static baseline features (same 70 features LightGBM ordering model uses)
    static = pd.read_parquet(DATA / "static.parquet")
    st_df = pd.DataFrame({"hadm_id": hadm}).merge(static, on="hadm_id", how="left")
    st_cols = [c for c in st_df.columns if c != "hadm_id"]
    S = st_df[st_cols].astype("float32").fillna(0.0).values
    static_dim = S.shape[1]
    print(f"Static baseline features: {static_dim}")

    # ---- identical subject-level split (seed 42) ----
    pat = lab.groupby("subject_id")["mortality"].max().reset_index().sample(frac=1, random_state=SEED)
    n = len(pat); n_tr, n_va = int(0.70 * n), int(0.15 * n)
    train_s = set(pat.iloc[:n_tr]["subject_id"]); val_s = set(pat.iloc[n_tr:n_tr + n_va]["subject_id"])
    s = lab["subject_id"].values
    tr = np.array([x in train_s for x in s]); va = np.array([x in val_s for x in s])
    te = ~tr & ~va
    assert train_s.isdisjoint(val_s) and train_s.isdisjoint(test_s := set(s[te])) and val_s.isdisjoint(test_s), \
        "subject overlap across splits"
    assert (tr & va).sum() == 0 and (tr & te).sum() == 0 and (va & te).sum() == 0
    assert tr.sum() + va.sum() + te.sum() == len(s), "split does not partition all rows"
    print(f"Split -- Train {tr.sum():,} | Val {va.sum():,} | Test {te.sum():,}")

    # ---- raw ordering-summary + static matrix for the in-script LightGBM comparator ----
    # (matches A_ordering_signal ordering_only: 3 per-channel summaries + 4 admission
    #  summaries per dataset + the same 70 static features; trees use RAW features)
    ord_lab_m, ord_lab_c = ordering_summary_features(obs_lab, ch_lab, "lab")
    ord_vit_m, ord_vit_c = ordering_summary_features(obs_vit, ch_vit, "vit")
    LGB_X = np.hstack([ord_lab_m, ord_vit_m, S]).astype("float32")   # S = RAW static
    print(f"LightGBM ordering comparator features: {LGB_X.shape[1]} "
          f"({len(ord_lab_c) + len(ord_vit_c)} ordering + {static_dim} static)")

    # ---- per-channel standardisation on TRAIN only ----
    mu = X[tr].reshape(-1, n_ch).mean(0)
    sd = X[tr].reshape(-1, n_ch).std(0); sd[sd == 0] = 1.0
    Xs = ((X - mu) / sd).astype("float32")
    Xt = torch.from_numpy(Xs).to(dev)                      # preload whole tensor to GPU
    # standardise static on TRAIN only
    smu = S[tr].mean(0); ssd = S[tr].std(0); ssd[ssd == 0] = 1.0
    St = torch.from_numpy(((S - smu) / ssd).astype("float32")).to(dev)

    # baseline LightGBM ordering AUROCs from experiment A
    a_base = {}
    if A_METRICS.exists():
        am = json.load(open(A_METRICS))["model_comparison"]["ordering_only"]
        a_base = {t: am[t]["test"]["auroc"] for t in TASKS}

    def run_variant(task, use_static, seed=SEED):
        torch.manual_seed(seed)                 # vary GRU init/shuffle per seed
        y_all = lab[task].values.astype("float32")
        ok = ~np.isnan(y_all)
        itr = np.where(tr & ok)[0]; iva = np.where(va & ok)[0]; ite = np.where(te & ok)[0]
        ytr = torch.from_numpy(y_all[itr]).to(dev)
        yva_np = y_all[iva]; yte_np = y_all[ite]
        prev = y_all[itr].mean()
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - prev) / prev, device=dev))
        sdim = static_dim if use_static else 0

        def fwd(model, idx):
            return model(Xt[idx], St[idx] if use_static else None)

        model = GRUClassifier(n_ch, args.hidden, args.layers, static_dim=sdim).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        itr_t = torch.from_numpy(itr).to(dev)
        best_auc, best_state, wait = -1, None, 0
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(itr), device=dev)
            for k in range(0, len(itr), args.batch):
                bi = itr_t[perm[k:k + args.batch]]
                opt.zero_grad()
                loss = crit(model(Xt[bi], St[bi] if use_static else None), ytr[perm[k:k + args.batch]])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(fwd(model, iva)).cpu().numpy()
            auc = roc_auc_score(yva_np, pv)
            if auc > best_auc:
                best_auc, best_state, wait = auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
            if wait >= args.patience:
                break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pte = torch.sigmoid(fwd(model, ite)).cpu().numpy()
        auroc = float(roc_auc_score(yte_np, pte)); auprc = float(average_precision_score(yte_np, pte))
        lo, hi = boot_ci(yte_np.astype(int), pte)
        metrics = {"auroc": round(auroc, 4), "auroc_ci": [round(lo, 4), round(hi, 4)],
                   "auprc": round(auprc, 4), "best_val_auroc": round(best_auc, 4)}
        return metrics, pte, ite, yte_np

    def train_lgbm_ordering(task):
        """Same-split, same-config LightGBM ordering_only (raw summaries + static).
        Returns matched test predictions on ite for a PAIRED comparison vs GRU.

        Early stopping monitors AUROC -- the same metric on which the GRU is
        model-selected -- so the rare-outcome models train to convergence and the
        GRU-vs-LightGBM comparison uses a single, consistent selection criterion."""
        y_all = lab[task].values.astype("float32")
        ok = ~np.isnan(y_all)
        itr = np.where(tr & ok)[0]; iva = np.where(va & ok)[0]; ite = np.where(te & ok)[0]
        prev = y_all[itr].mean()
        m = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=127, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            scale_pos_weight=(1 - prev) / prev, metric="auc",
            n_jobs=-1, random_state=SEED, verbose=-1)
        m.fit(LGB_X[itr], y_all[itr], eval_set=[(LGB_X[iva], y_all[iva])],
              eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        pte = m.predict_proba(LGB_X[ite])[:, 1]
        return pte, ite, float(roc_auc_score(y_all[ite], pte))

    results = {}
    for task in args.tasks:
        a_scalar = a_base.get(task)                       # cross-check vs experiment A
        lgb_pte, lgb_ite, lgb_auroc = train_lgbm_ordering(task)      # deterministic comparator
        co_m, _, _, _ = run_variant(task, use_static=False, seed=SEED)   # GRU counts only (1 seed)

        # GRU(counts+static) across multiple seeds (codex #6: not a single stochastic seed)
        seed_aurocs, seed_deltas, seed_ps = [], [], []
        for sd in args.gru_seeds:
            cs_m, cs_pte, cs_ite, cs_yte = run_variant(task, use_static=True, seed=sd)
            assert np.array_equal(cs_ite, lgb_ite), "GRU/LightGBM test rows differ"
            d, lo, hi, pval = paired_boot(cs_yte.astype(int), cs_pte, lgb_pte)
            seed_aurocs.append(cs_m["auroc"]); seed_deltas.append(d); seed_ps.append(pval)
        sa = np.array(seed_aurocs); sdlt = np.array(seed_deltas)
        results[task] = {
            "gru_counts_only_seed42": co_m,
            "gru_counts_plus_static_seeds": args.gru_seeds,
            "gru_cs_auroc_mean": round(float(sa.mean()), 4),
            "gru_cs_auroc_sd": round(float(sa.std(ddof=1)), 4),
            "gru_cs_auroc_per_seed": [round(x, 4) for x in seed_aurocs],
            "lightgbm_ordering_inscript_auroc": round(lgb_auroc, 4),
            "lightgbm_ordering_A_scalar_crosscheck": a_scalar,
            "paired_delta_mean": round(float(sdlt.mean()), 4),
            "paired_delta_sd": round(float(sdlt.std(ddof=1)), 4),
            "paired_delta_per_seed": [round(x, 4) for x in seed_deltas],
            "paired_p_per_seed": [round(x, 4) for x in seed_ps],
        }
        print(f"  >> {task}: GRU(c+s) {sa.mean():.4f}±{sa.std(ddof=1):.4f} "
              f"(seeds {seed_aurocs}) | LGBM {lgb_auroc:.4f} (A={a_scalar}) "
              f"| paired d={sdlt.mean():+.4f}±{sdlt.std(ddof=1):.4f}")
        json.dump(results, open(OUT / "gru_metrics.json", "w"), indent=2)

    print(f"\n==== GRU(counts+static) vs in-script LightGBM ordering (paired, "
          f"{len(args.gru_seeds)} GRU seeds) ====")
    print(f"  {'task':16}{'GRU c+s (mean±sd)':>22}{'LGBM':>9}{'delta (mean±sd)':>20}")
    for t, r in results.items():
        print(f"  {t:16}{r['gru_cs_auroc_mean']:>13.4f}±{r['gru_cs_auroc_sd']:<8.4f}"
              f"{r['lightgbm_ordering_inscript_auroc']:>9.4f}"
              f"{r['paired_delta_mean']:>+12.4f}±{r['paired_delta_sd']:<7.4f}")
    print(f"\nSaved -> {OUT}/gru_metrics.json")


if __name__ == "__main__":
    main()
