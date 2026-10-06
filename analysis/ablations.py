"""Reproduce the validation results in the README.

5-fold CV (KFold, shuffle, random_state=0) scored three ways:
  * complete rows      - training rows with no missing inputs
  * all train rows     - every training row, as given
  * test-like masked   - every validation row re-masked with the missing-value patterns found in test.csv
                         (sampled from their empirical frequencies), so the score reflects the test set's
                         ~7%-per-column missingness instead of train's ~1.5%

Usage (from the repository root, with train.csv and test.csv from the datathon placed there):
    python analysis/ablations.py
Runtime: a few minutes on a laptop (fits run in parallel).
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import energy_model as em  # noqa: E402

train = pd.read_csv(ROOT / "train.csv")
test = pd.read_csv(ROOT / "test.csv")
y = train["energy_usage"].values
NUM, RAW = em.NUM, em.RAW_FEATURES

# validation rows re-masked with the test set's own missing-value patterns
pats, cnt = np.unique(test[NUM].isna().values, axis=0, return_counts=True)
val_mask = pats[np.random.default_rng(100).choice(len(pats), size=len(train), p=cnt / cnt.sum())]
train_masked = train.copy()
for j, c in enumerate(NUM):
    train_masked.loc[val_mask[:, j], c] = np.nan
keep = pats.sum(1) > 0
mask_patterns = (pats[keep], cnt[keep] / cnt[keep].sum())

# configuration of the submitted model (model.pkl)
FINAL = dict(alpha=2.0, w_lgb=0.0,
             missing_offsets={"previous_usage": 1.6, "occupancy": 0.3, "humidity": 0.25, "temperature": -0.25})
EXTRA = dict(lin_imp_w=1.0, imp_alpha=3.0, fit_all_rows=True, design_opts={"month_sincos": 1},
             wls=True, wls_c=20.0, wls_pow=4.0)

# each ablation changes exactly one decision of the final model
ABLATIONS = {
    "Final model (submitted)": ({}, {}),
    "without missing-value offsets": ({"missing_offsets": {}}, {}),
    "without weighted least squares": ({}, {"wls": False}),
    "LightGBM imputers instead of ridge imputers": ({}, {"lin_imp_w": 0.0}),
    "linear model fit on complete rows only": ({}, {"fit_all_rows": False}),
    "+15% LightGBM blend": ({"w_lgb": 0.15}, {}),
    "building x hour profiles instead of type x hour": ({}, {"design_opts": {"month_sincos": 1, "bh": True}}),
    "without cooling term above 28C": ({}, {"design_opts": {"month_sincos": 1, "no_hinge": True}}),
    "without previous_usage": ({}, {"design_opts": {"month_sincos": 1, "no_prev": True}}),
}
FOLDS = list(KFold(5, shuffle=True, random_state=0).split(train))


def design_flex(df, buildings, b_type, opts=None):
    """The model's design matrix with optional blocks swapped out (for the design ablations)."""
    opts = opts or {}
    Z = em._design_original(df, buildings, b_type, opts)
    n, nb, nt = len(df), len(buildings), int(np.max(b_type)) + 1
    # block layout: building, type, hour, type x hour, building x weekend, day of week, month terms,
    # type x temperature, building x humidity, building x occupancy, previous_usage, previous_usage^2, hinge
    sizes = [nb, nt, 24, nt * 24, nb, 7, 4 if opts.get("month_sincos") else 12, nt, nb, nb, 1, 1, 1]
    edges = np.cumsum([0] + sizes)
    blocks = [Z[:, edges[i]:edges[i + 1]] for i in range(len(sizes))]
    if opts.get("bh"):
        BH = np.zeros((n, nb * 24))
        BH[np.arange(n), df["_b"].values * 24 + df["hour"].values.astype(int)] = 1
        blocks[3] = BH
    if opts.get("no_hinge"):
        blocks[12] = blocks[12][:, :0]
    if opts.get("no_prev"):
        blocks[10], blocks[11] = blocks[10][:, :0], blocks[11][:, :0]
    return np.hstack(blocks)


def fit_ablation(name, k):
    if not hasattr(em, "_design_original"):          # patch once per worker process
        em._design_original = em.design
        em.design = design_flex
    final_over, extra_over = ABLATIONS[name]
    model = em.EnergyModel(seeds=(0,), mask_patterns=mask_patterns, **{**FINAL, **final_over})
    for key, value in {**EXTRA, **extra_over}.items():
        setattr(model, key, value)
    a, b = FOLDS[k]
    model.fit(train.iloc[a][RAW], y[a], unlabeled=test[RAW])        # imputers also see test features (no labels)
    return name, b, model.predict(train.iloc[b][RAW]), model.predict(train_masked.iloc[b][RAW])


def fit_lightgbm(k):
    def raw_X(df):
        X = df[RAW].copy()
        for c in ["building_id", "building_type", "day_of_week"]:
            X[c] = X[c].astype(pd.CategoricalDtype(sorted(train[c].unique())))
        return X
    a, b = FOLDS[k]
    m = lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=15, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, verbose=-1).fit(raw_X(train.iloc[a]), y[a])
    return "LightGBM on raw features (baseline)", b, m.predict(raw_X(train.iloc[b])), m.predict(raw_X(train_masked.iloc[b]))


def main():
    jobs = [delayed(fit_ablation)(n, k) for n in ABLATIONS for k in range(5)] + [delayed(fit_lightgbm)(k) for k in range(5)]
    out = Parallel(n_jobs=-1)(jobs)
    complete = train[NUM].notna().all(axis=1).values
    rmse = lambda p, s=slice(None): float(np.sqrt(np.mean((p[s] - y[s]) ** 2)))
    oof = {}
    for name, b, p, pm in out:
        o = oof.setdefault(name, (np.zeros(len(y)), np.zeros(len(y))))
        o[0][b], o[1][b] = p, pm
    rows = {n: {"complete rows": rmse(p, complete), "all train rows": rmse(p), "test-like masked": rmse(pm)}
            for n, (p, pm) in oof.items()}
    print("## Cross-validated RMSE (5-fold)\n")
    cols = ["complete rows", "all train rows", "test-like masked"]
    print("| Configuration | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for n, r in rows.items():
        print(f"| {n} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")

    p0 = oof["without missing-value offsets"][0]
    natural = train[NUM].isna().values
    print("\n## Are values missing at random? Out-of-fold mean(actual - predicted) on rows naturally missing each input\n")
    for j, c in enumerate(NUM):
        r = y[natural[:, j]] - p0[natural[:, j]]
        print(f"- {c}: n={natural[:, j].sum()}, {r.mean():+.2f} ± {r.std(ddof=1) / np.sqrt(len(r)):.2f} (SE); "
              f"offset used: {FINAL['missing_offsets'][c]:+.2f}")

    p, pm = oof["Final model (submitted)"]
    res = y[complete] - p[complete]
    print(f"\n## Residuals of the final model (complete rows)\n\nsd {res.std():.3f}, skew {stats.skew(res):.2f}, "
          f"excess kurtosis {stats.kurtosis(res):.2f}")
    print("\n## Final model, test-like error by missing input\n")
    m = train_masked[NUM].isna().values
    for j, c in enumerate(NUM):
        print(f"- {c} missing: {rmse(pm, m[:, j]):.2f} (n={m[:, j].sum()})")
    print(f"- nothing missing: {rmse(pm, ~m.any(1)):.2f} (n={(~m.any(1)).sum()})")


if __name__ == "__main__":
    main()
