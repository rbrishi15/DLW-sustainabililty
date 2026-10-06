"""Track 1 - Smart Campus energy model.

Submitted configuration = ImputeLinear only (the LightGBM component is trained but given zero weight, w_lgb = 0).

ImputeLinear: the energy data behaves like an additive linear process:
    energy ~ building-type x hour profile + building base load and weekend shift
             + day-of-week and seasonal terms
             + slopes on occupancy and humidity (per building) and temperature (per type)
             + cooling hinge above 28C + ~0.34 * previous_usage
A ridge regression on that design gets CV RMSE ~3.02 on complete rows (LightGBM ~3.22).
Because the target is linear in the numeric inputs, a missing input is best
replaced by its conditional mean. The submitted model uses ridge imputers (one per
missing column and set of other columns present, lin_imp_w = 1.0); LightGBM imputers
trained on the other columns (with random extra masking) are also fitted but unused.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

NUM = ["temperature", "humidity", "occupancy", "previous_usage"]
CAT = ["building_id", "building_type", "hour", "day_of_week", "month"]
RAW_FEATURES = CAT + NUM
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DOW = {d: i for i, d in enumerate(DAYS)}
CENTER = {"temperature": 28.0, "humidity": 78.0, "occupancy": 70.0, "previous_usage": 50.0}


def design(df, buildings, b_type, opts=None):
    opts = opts or {}
    """Linear design matrix. df must be sanitized; b_type maps building index -> building-type index.

    CV showed hourly profiles are shared by buildings of the same type (type x hour beats
    building x hour), while base load, weekend shift and occupancy/humidity slopes are per building.
    """
    n = len(df)
    r = np.arange(n)
    nb = len(buildings)
    nt = int(np.max(b_type)) + 1
    b = df["_b"].values
    t = np.asarray(b_type)[b]
    h = df["hour"].values.astype(int)
    dw = df["_dow"].values
    we = (dw >= 5).astype(float)
    mo = df["month"].values.astype(int)
    Bm = np.zeros((n, nb)); Bm[r, b] = 1
    Tm = np.zeros((n, nt)); Tm[r, t] = 1
    Hm = np.zeros((n, 24)); Hm[r, h] = 1
    TH = np.zeros((n, nt * 24)); TH[r, t * 24 + h] = 1
    D = np.zeros((n, 7)); D[r, dw] = 1
    M = np.zeros((n, 12)); M[r, mo - 1] = 1
    if opts.get("harmonic"):
        hh = 2 * np.pi * h / 24
        HS = np.c_[np.sin(hh), np.cos(hh), np.sin(2 * hh), np.cos(2 * hh)]
        if opts.get("harmonic") >= 3:
            HS = np.c_[HS, np.sin(3 * hh), np.cos(3 * hh)]
        ang = 2 * np.pi * mo / 12
        Ms = np.c_[np.sin(ang), np.cos(ang)]
        occ = df["occupancy"].values.astype(float); tmp = df["temperature"].values.astype(float)
        cols = [Bm, Tm, D, Ms, HS, Tm * we[:, None], (Tm[:, :, None] * HS[:, None, :]).reshape(n, -1),
                Tm * occ[:, None], Tm * tmp[:, None], occ[:, None], tmp[:, None],
                df["humidity"].values.astype(float)[:, None], df["previous_usage"].values.astype(float)[:, None]]
        if opts.get("keep_extras"):
            cols += [np.maximum(tmp - 28.0, 0)[:, None], Bm * (occ - CENTER["occupancy"])[:, None]]
        return np.hstack(cols)
    if opts.get("month_sincos"):
        ang = 2 * np.pi * (mo - 1) / 12
        M = np.c_[np.sin(ang), np.cos(ang)] if opts.get("month_sincos") == 1 and opts.get("one_harmonic") else np.c_[np.sin(ang), np.cos(ang), np.sin(2 * ang), np.cos(2 * ang)]
    cols = [Bm, Tm, Hm, TH, Bm * we[:, None], D, M]
    if opts.get("type_occ"):
        cols.append(Tm * (df["occupancy"].values - CENTER["occupancy"])[:, None])
    for c in NUM:
        v = df[c].values - CENTER[c]
        if c == "previous_usage":
            cols += [v[:, None], (v ** 2 / 100.0)[:, None]]
        elif c == "temperature":
            cols.append(Tm * v[:, None])
        else:
            cols.append(Bm * v[:, None])
    # extra cooling load once it gets hot (hinge at 28C)
    cols.append(np.maximum(df["temperature"].values - 28.0, 0)[:, None])
    return np.hstack(cols)


def imp_design(df, others, nb):
    """Linear design for the imputers (building x hour profiles, month x hour, slopes on the other columns)."""
    n = len(df); r = np.arange(n)
    b = df["_b"].values; h = df["hour"].values.astype(int); dw = df["_dow"].values
    we = (dw >= 5).astype(float); mo = df["month"].values.astype(int) - 1
    def oh(idx, k):
        M = np.zeros((n, k)); M[r, idx] = 1; return M
    Bm = oh(b, nb); BH = oh(b * 24 + h, nb * 24)
    cols = [Bm, oh(h, 24), oh(h, 24) * we[:, None], BH, BH * we[:, None], Bm * we[:, None], oh(dw, 7), oh(mo, 12), oh(mo * 24 + h, 288) * 0.3]
    for c in others:
        v = df[c].values - CENTER[c]
        cols += [Bm * v[:, None], BH * v[:, None] * 0.1]
    return np.hstack(cols)


class EnergyModel:
    def __init__(self, alpha=3.0, w_lgb=0.15, seeds=(0, 1, 2), aug_rate=0.1, mask_patterns=None, missing_offsets=None):
        self.alpha, self.w_lgb, self.seeds, self.aug_rate = alpha, w_lgb, seeds, aug_rate
        # Values are not missing at random: rows whose value went missing tend to be unusual (events,
        # equipment left on), so their usage sits above the imputed prediction. Per-column offsets are
        # estimated out-of-fold on train rows with naturally missing values (then shrunk) and added here.
        self.missing_offsets = dict(missing_offsets or {})
        # (patterns, probabilities) of which numeric columns are missing together; used to mask
        # training copies for the LightGBM part. Default: each column missing independently 7%.
        if mask_patterns is None:
            pats = np.array([[(k >> j) & 1 for j in range(4)] for k in range(1, 16)], bool)
            pr = np.array([0.07 ** p.sum() * 0.93 ** (4 - p.sum()) for p in pats])
            mask_patterns = (pats, pr / pr.sum())
        self.mask_patterns = mask_patterns
        self.imp_params = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1)
        self.lgb_params = dict(n_estimators=800, learning_rate=0.035, num_leaves=15, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)

    # ---------- input cleaning (robust to unseen categories / out-of-range values)
    def _sanitize(self, df):
        df = df.copy()
        for c in NUM + ["hour", "month"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        for c in ["building_id", "building_type", "day_of_week"]:
            if c not in df.columns:
                df[c] = np.nan
            df[c] = df[c].astype("string").str.strip()
        df["day_of_week"] = df["day_of_week"].str.capitalize()
        # building: unknown id -> first known building of the same type -> most common building
        bmap = {k: i for i, k in enumerate(self.buildings_)}
        b = df["building_id"].map(bmap)
        by_type = df["building_type"].map(self.type_to_b_)
        b = b.fillna(by_type).fillna(self.b_mode_)
        df["_b"] = b.astype(int).values
        df["_dow"] = df["day_of_week"].map(DOW).fillna(self.dow_mode_).astype(int).values
        df["hour"] = df["hour"].fillna(self.hour_mode_).round().clip(0, 23)
        df["month"] = df["month"].fillna(self.month_mode_).round().clip(1, 12)
        for c in NUM:  # keep values inside a widened training range so the linear part can't explode
            lo, hi = self.ranges_[c]
            df[c] = df[c].clip(lo, hi)
        return df

    def _imp_X(self, df):
        X = pd.DataFrame({"b": df["_b"].values, "hour": df["hour"].values, "dow": df["_dow"].values,
                          "month": df["month"].values})
        for c in NUM:
            X[c] = df[c].values
        return X

    # ---------- training
    def fit(self, df, y, unlabeled=None):
        """unlabeled: optional extra rows (no target, e.g. test features) used only to train the imputers."""
        df = df.reset_index(drop=True)
        y = np.asarray(y, float)
        self.buildings_ = sorted(df["building_id"].dropna().astype(str).str.strip().unique())
        bmap = {k: i for i, k in enumerate(self.buildings_)}
        tmp = df.dropna(subset=["building_id", "building_type"])
        self.type_to_b_ = {t: bmap[g["building_id"].astype(str).str.strip().mode()[0]]
                           for t, g in tmp.groupby(tmp["building_type"].astype(str).str.strip())}
        self.b_mode_ = bmap[df["building_id"].astype(str).str.strip().mode()[0]]
        types = sorted(tmp["building_type"].astype(str).str.strip().unique())
        b2t = tmp.assign(_bid=tmp["building_id"].astype(str).str.strip(), _t=tmp["building_type"].astype(str).str.strip())
        b2t = b2t.groupby("_bid")["_t"].agg(lambda x: x.mode()[0])
        self.b_type_ = np.array([types.index(b2t[k]) if k in b2t.index else 0 for k in self.buildings_])
        self.dow_mode_ = DOW[df["day_of_week"].mode()[0]]
        self.hour_mode_ = float(df["hour"].median())
        self.month_mode_ = float(df["month"].median())
        self.ranges_ = {}
        for c in NUM:
            lo, hi = float(df[c].min()), float(df[c].max())
            pad = 0.25 * (hi - lo)
            self.ranges_[c] = (max(lo - pad, 0.0), hi + pad)
        self.ranges_["humidity"] = (self.ranges_["humidity"][0], 100.0)
        self.y_max_ = float(y.max())
        d = self._sanitize(df)
        X = self._imp_X(d)
        X_imp = X if unlabeled is None else pd.concat(
            [X, self._imp_X(self._sanitize(pd.DataFrame(unlabeled).reset_index(drop=True)))], ignore_index=True)

        # 1) imputers: E[column | everything else], bagged over seeds
        self.imputers_ = {c: [] for c in NUM}
        for s in self.seeds:
            rng = np.random.default_rng(s)
            for c in NUM:
                ok = X_imp[c].notna().values
                Xi = X_imp.loc[ok].drop(columns=[c])
                reps = [Xi]
                for _ in range(2):
                    Xa = Xi.copy()
                    for o in NUM:
                        if o != c:
                            Xa.loc[rng.random(len(Xa)) < 2 * self.aug_rate, o] = np.nan
                    reps.append(Xa)
                m = lgb.LGBMRegressor(**self.imp_params, random_state=s)
                m.fit(pd.concat(reps, ignore_index=True), np.tile(X_imp.loc[ok, c].values, 3), categorical_feature=["b"])
                self.imputers_[c].append(m)

        # 1b) linear imputers, one ridge per (column, set of other columns present)
        import itertools
        d_imp = d if unlabeled is None else pd.concat([d, self._sanitize(pd.DataFrame(unlabeled).reset_index(drop=True))], ignore_index=True)
        self.lin_imputers_ = {}
        ia = getattr(self, "imp_alpha", 3.0)
        for c in NUM:
            rest = [o for o in NUM if o != c]
            for k in range(len(rest) + 1):
                for avail in itertools.combinations(rest, k):
                    ok = d_imp[[c] + list(avail)].notna().all(axis=1).values
                    self.lin_imputers_[(c, frozenset(avail))] = Ridge(alpha=ia).fit(
                        imp_design(d_imp[ok], list(avail), len(self.buildings_)), d_imp.loc[ok, c].values)

        # 2) linear model on complete rows (optionally all rows, imputed)
        cc = d[NUM].notna().all(axis=1).values
        if getattr(self, "fit_all_rows", False):
            Zd = design(self.impute(d), self.buildings_, self.b_type_, getattr(self, "design_opts", None))
            self.linear_ = Ridge(alpha=self.alpha).fit(Zd, y)
            if getattr(self, "wls", False):
                sw = 1.0 / (np.maximum(self.linear_.predict(Zd), 10) + getattr(self, "wls_c", 40.0)) ** getattr(self, "wls_pow", 1.0); sw = sw / sw.mean()
                self.linear_ = Ridge(alpha=self.alpha).fit(Zd, y, sample_weight=sw)
        else:
            self.linear_ = Ridge(alpha=self.alpha).fit(design(d[cc], self.buildings_, self.b_type_, getattr(self, "design_opts", None)), y[cc])

        # 3) LightGBM on raw features, trained with test-like random masking
        self.mask_p_ = self.mask_patterns
        self.lgbs_ = []
        for s in self.seeds:
            rng = np.random.default_rng(1000 + s)
            parts, ys = [X], [y]
            for _ in range(2):
                Xa = X.copy()
                pats = self.mask_p_[0][rng.choice(len(self.mask_p_[1]), size=len(X), p=self.mask_p_[1])]
                for j, c in enumerate(NUM):
                    Xa.loc[pats[:, j], c] = np.nan
                parts.append(Xa); ys.append(y)
            m = lgb.LGBMRegressor(**self.lgb_params, random_state=s)
            m.fit(pd.concat(parts, ignore_index=True), np.concatenate(ys), categorical_feature=["b"])
            self.lgbs_.append(m)
        return self

    # ---------- inference
    def impute(self, d):
        d = d.copy()
        X = self._imp_X(d)
        orig = d[NUM].notna()
        w = getattr(self, "lin_imp_w", 1.0)
        for c in NUM:
            miss = X[c].isna().values
            if not miss.any():
                continue
            v = np.mean([m.predict(X.loc[miss].drop(columns=[c])) for m in self.imputers_[c]], axis=0)
            if w > 0 and hasattr(self, "lin_imputers_"):
                rest = [o for o in NUM if o != c]
                sub = d.loc[miss]
                keys = [frozenset(o for o in rest if orig.loc[i, o]) for i in sub.index]
                lin = np.zeros(len(sub))
                for key in set(keys):
                    sel = np.array([k == key for k in keys])
                    lin[sel] = self.lin_imputers_[(c, key)].predict(imp_design(sub.loc[sel], [o for o in NUM if o in key], len(self.buildings_)))
                v = (1 - w) * v + w * lin
            d.loc[miss, c] = v
        return d

    def predict_parts(self, df):
        d = self._sanitize(pd.DataFrame(df).reset_index(drop=True))
        p_lin = self.linear_.predict(design(self.impute(d), self.buildings_, self.b_type_, getattr(self, "design_opts", None)))
        p_lgb = np.mean([m.predict(self._imp_X(d)) for m in self.lgbs_], axis=0)
        return p_lin, p_lgb, d[NUM].isna().values

    def predict(self, df):
        d = self._sanitize(pd.DataFrame(df).reset_index(drop=True))
        p_lin = self.linear_.predict(design(self.impute(d), self.buildings_, self.b_type_, getattr(self, "design_opts", None)))
        X = self._imp_X(d)
        p_lgb = np.mean([m.predict(X) for m in self.lgbs_], axis=0)
        p = (1 - self.w_lgb) * p_lin + self.w_lgb * p_lgb
        for c, off in getattr(self, "missing_offsets", {}).items():
            p = p + off * d[c].isna().values
        return np.clip(p, 0.0, 1.5 * self.y_max_)


class BlendModel:
    """Weighted average of several fitted models (each with .predict on the raw feature frame)."""

    def __init__(self, models, weights):
        self.models, self.weights = list(models), list(weights)

    def predict(self, df):
        return sum(w * m.predict(df) for m, w in zip(self.models, self.weights))


class MissingBlend:
    """Use `base` everywhere; on rows with any missing numeric input, blend in `fallback` with weight w."""

    def __init__(self, base, fallback, w):
        self.base, self.fallback, self.w = base, fallback, w

    def predict(self, df):
        df = pd.DataFrame(df).reset_index(drop=True)
        p = np.asarray(self.base.predict(df), float).copy()
        miss = df[NUM].apply(pd.to_numeric, errors="coerce").isna().any(axis=1).values
        if miss.any():
            p[miss] = (1 - self.w) * p[miss] + self.w * np.asarray(self.fallback.predict(df.loc[miss]), float)
        return p


class OffsetModel:
    """Adds a constant to predictions for rows where a given column is missing."""

    def __init__(self, base, offsets):
        self.base, self.offsets = base, dict(offsets)

    def predict(self, df):
        df = pd.DataFrame(df).reset_index(drop=True)
        p = np.asarray(self.base.predict(df), float).copy()
        for c, o in self.offsets.items():
            p = p + o * pd.to_numeric(df[c], errors="coerce").isna().values
        return p
