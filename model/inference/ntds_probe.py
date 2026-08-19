"""Probe ntds gait embeddings for age / BMI / gender, and render result plots.

Thin wrapper over `model/training/utils/linear_probe.py` (subject-level cross-validated
Ridge / LogisticRegression, per-activity + ensemble). Plain Python — the notebook imports
it to show results inline, and it also runs headless to write PNGs into `results/`.
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from model.training.utils.linear_probe import (
    run_cross_validated_ridge_probe as _probe1,
    run_cross_validated_ridge_probe_with_ensemble as _probeE,
)

# (label_key_in_df, display_name, task_type, unit)
TARGETS = [
    ("age", "Age", "reg", "yr"),
    ("bmi", "BMI", "reg", "kg/m²"),
    ("gender_male", "Gender", "clas", ""),
]


def _matrix(emb_df: pd.DataFrame):
    E = emb_df[[c for c in emb_df.columns if c.startswith("emb_")]].values.astype(np.float64)
    E = E[:, E.std(axis=0) > 1e-8]          # drop constant dims (no signal; break standardization)
    ids = emb_df["participant_id"].tolist()
    acts = emb_df["activity"].tolist()
    labels = [emb_df["age"].values, emb_df["bmi"].values, emb_df["gender_male"].values]
    return E, ids, acts, labels


# Apple-Accelerate BLAS emits spurious FP-flag RuntimeWarnings during matmul (X.T @ X);
# harmless, results are unaffected. Run every probe call with those numpy FP warnings silenced.
_FP_IGNORE = dict(divide="ignore", over="ignore", invalid="ignore")


def _E(*a, **k):
    with np.errstate(**_FP_IGNORE):
        return _probeE(*a, **k)


def _1(*a, **k):
    with np.errstate(**_FP_IGNORE):
        return _probe1(*a, **k)


def probe_all(emb_df: pd.DataFrame, n_folds: int = 5, alpha: float = 1.0,
              targets: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Return a tidy metrics table (target × activity/ensemble → pearson/r2/mae or auc/acc).

    `targets` optionally restricts which display names to run (e.g. ["Age", "BMI"]).
    """
    E, ids, acts, labels = _matrix(emb_df)
    rows = []
    for idx, (_key, name, task, _unit) in enumerate(TARGETS):
        if targets is not None and name not in targets:
            continue
        res = _E(E, labels, ids, acts, idx, task_type=task, n_folds=n_folds, alpha=alpha)
        if not res:
            continue
        for activity, mets in res.items():
            row = {"target": name, "activity": activity}
            for k in ("pearson", "r2", "mae", "auc", "accuracy"):
                if k in mets:
                    row[k] = round(float(mets[k]), 4)
            rows.append(row)
    return pd.DataFrame(rows)


def probe_all_stratified(emb_df: pd.DataFrame, n_folds: int = 5, alpha: float = 1.0) -> pd.DataFrame:
    """Paper-style: age & BMI probed WITHIN each sex separately; gender probed un-stratified.

    Returns a tidy table with a `sex` column ('male'/'female'/'all').
    """
    out = []
    gender = probe_all(emb_df, n_folds=n_folds, alpha=alpha)
    for _, r in gender[gender["target"] == "Gender"].iterrows():
        out.append({"target": "Gender", "sex": "all", "activity": r["activity"],
                    "auc": r.get("auc"), "accuracy": r.get("accuracy")})
    for sex_val, sex_name in ((1.0, "male"), (0.0, "female")):
        sub = emb_df[emb_df["gender_male"] == sex_val]
        m = probe_all(sub, n_folds=n_folds, alpha=alpha, targets=["Age", "BMI"])
        for tgt in ("Age", "BMI"):
            for _, r in m[m["target"] == tgt].iterrows():
                out.append({"target": tgt, "sex": sex_name, "activity": r["activity"],
                            "pearson": r.get("pearson"), "r2": r.get("r2"), "mae": r.get("mae")})
    return pd.DataFrame(out)


def _ensemble_gender_predictions(E, ids, acts, labels, n_folds=5, alpha=1.0):
    """Reconstruct per-subject ensemble gender probabilities (averaged over activities)."""
    per, true = defaultdict(dict), {}
    A = set(acts)
    for act in A:
        res = _1(E, labels, ids, acts, 2, task_type="clas", activity_to_run=act,
                 n_folds=n_folds, alpha=alpha)
        if res is None:
            continue
        p = res["predictions"]
        for uid, yt, yp in zip(p["unique_ids"], p["y_test"], p["y_pred"]):
            per[uid][act] = yp
            true[uid] = yt
    keep = [u for u in per if len(per[u]) == len(A)]
    y_true = np.array([true[u] for u in keep]).astype(int)
    y_proba = np.array([np.mean(list(per[u].values())) for u in keep])
    return y_true, y_proba


def make_plots(emb_df: pd.DataFrame, out_dir: str = "results",
               prefix: str = "ntds_gait_smoke", n_folds: int = 5, alpha: float = 1.0):
    """Write age scatter, BMI scatter, and gender ROC PNGs. Returns {name: path}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    os.makedirs(out_dir, exist_ok=True)
    E, ids, acts, labels = _matrix(emb_df)
    paths = {}

    for idx, color in ((0, "#2c7fb8"), (1, "#1a9e3a")):          # age, bmi
        _key, name, _task, unit = TARGETS[idx]
        res = _E(E, labels, ids, acts, idx, task_type="reg", n_folds=n_folds, alpha=alpha)
        p = res["ensemble"]["predictions"]
        yt, yp = np.array(p["y_test"]), np.array(p["y_pred"])
        r, r2, mae = res["ensemble"]["pearson"], res["ensemble"]["r2"], res["ensemble"]["mae"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(yt, yp, s=16, alpha=0.6, color=color, edgecolor="none")
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lim, lim, "--", color="0.6", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"True {name} ({unit})"); ax.set_ylabel(f"Predicted {name} ({unit})")
        ax.set_title(f"{name}: r={r:.2f}, R²={r2:.2f}, MAE={mae:.2f} {unit}\n"
                     f"gait ensemble · subject-level {n_folds}-fold CV · n={len(yt)}", fontsize=10)
        fig.tight_layout()
        path = os.path.join(out_dir, f"{prefix}_{name.lower()}_scatter.png")
        fig.savefig(path, dpi=130); plt.close(fig)
        paths[name.lower()] = path

    y_true, y_proba = _ensemble_gender_predictions(E, ids, acts, labels, n_folds, alpha)
    fpr, tpr, _ = roc_curve(y_true, y_proba); auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.6", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Gender (male vs female)\ngait ensemble · subject-level {n_folds}-fold CV · n={len(y_true)}",
                 fontsize=10)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_gender_roc.png")
    fig.savefig(path, dpi=130); plt.close(fig)
    paths["gender"] = path
    return paths


# =============================================================================
# Paper-faithful nested cross-validation (outer 5-fold, inner 4-fold α-tuning, ×N seeds).
# One model over all clips (features = 1024-d embedding + activity one-hot), split by
# participant; per-subject prediction = mean over that subject's clips (late fusion).
# =============================================================================

def _feature_matrix(emb_df: pd.DataFrame) -> np.ndarray:
    E = emb_df[[c for c in emb_df.columns if c.startswith("emb_")]].values.astype(np.float64)
    E = E[:, E.std(axis=0) > 1e-8]
    acts = pd.get_dummies(emb_df["activity"].astype(str)).values.astype(np.float64)
    return np.hstack([E, acts])


def _participant_folds(groups: np.ndarray, k: int, seed: int):
    """Yield (train_idx, test_idx) with every participant confined to one fold."""
    rng = np.random.RandomState(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    rng.shuffle(uniq)
    fold_of = {g: fi for fi, part in enumerate(np.array_split(uniq, k)) for g in part}
    gi = np.array([fold_of[g] for g in groups])
    for fi in range(k):
        yield np.where(gi != fi)[0], np.where(gi == fi)[0]


def _score(y_true, y_pred, task):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true.astype(int), y_pred) if task == "clas" else pearsonr(y_true, y_pred)[0]


# Non-linear GBDT candidate (sklearn's histogram GBM — a LightGBM-equivalent that needs no
# libomp/Homebrew). Added to the inner-loop model selection for REGRESSION targets only;
# gender is already ~perfect with the linear model, so GBDT is not a candidate there.
_HGB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=0)

# The GBDT was evaluated against Ridge in the inner loop and is NEVER selected on the v5
# embeddings (Ridge wins every fold, by ~0.03-0.07). Off by default so the probe stays fast;
# set True to re-include it in the competitive selection (much slower, identical results).
_USE_GBDT = False


def _candidates(task, grid):
    lin = [("lin", a) for a in grid]
    return lin + ([("hgb", _HGB)] if (task == "reg" and _USE_GBDT) else [])


def _fit_predict(Xtr, ytr, Xte, cfg, task):
    kind, hp = cfg
    if kind == "lin":
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        if task == "clas":
            return LogisticRegression(C=1.0 / hp, max_iter=2000).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        return Ridge(alpha=hp).fit(Xtr, ytr).predict(Xte)
    return HistGradientBoostingRegressor(**hp).fit(Xtr, ytr).predict(Xte)   # tree model — no scaling


def _pick_model(X, y, groups, inner, grid, seed, task):
    """Inner-CV selection over Ridge/LogReg grid + (for regression) the GBDT — returns best cfg."""
    best, best_score = None, -np.inf
    for cfg in _candidates(task, grid):
        scores = [_score(y[ite], _fit_predict(X[itr], y[itr], X[ite], cfg, task), task)
                  for itr, ite in _participant_folds(groups, inner, seed + 101)]
        ms = np.nanmean(scores)
        if ms > best_score:
            best_score, best = ms, cfg
    return best


def _nested(emb_df, ycol, task, n_seeds=15, outer=5, inner=4,
            grid=(0.1, 1.0, 10.0, 100.0, 1000.0)):
    """Return (mean, std, (y_true, y_pred) from last seed) over n_seeds nested-CV repeats."""
    from collections import defaultdict, Counter
    X = _feature_matrix(emb_df)
    y = emb_df[ycol].values.astype(float)
    groups = emb_df["participant_id"].values
    metrics, last, sel = [], None, Counter()
    with np.errstate(**_FP_IGNORE):
        for seed in range(n_seeds):
            ps_pred, ps_true = defaultdict(list), {}
            for tr, te in _participant_folds(groups, outer, seed):
                cfg = _pick_model(X[tr], y[tr], groups[tr], inner, grid, seed, task)
                sel[cfg[0]] += 1
                pred = _fit_predict(X[tr], y[tr], X[te], cfg, task)
                for i, p in zip(te, pred):
                    ps_pred[groups[i]].append(p); ps_true[groups[i]] = y[i]
            ids = list(ps_pred)
            yp = np.array([np.mean(ps_pred[s]) for s in ids])
            yt = np.array([ps_true[s] for s in ids])
            metrics.append(_score(yt, yp, task))
            last = (yt, yp)
    print(f"    [{ycol}/{task}] model picks: {dict(sel)}", flush=True)   # lin vs hgb across folds
    return float(np.nanmean(metrics)), float(np.nanstd(metrics)), last


def nested_probe_stratified(emb_df: pd.DataFrame, n_seeds: int = 15) -> pd.DataFrame:
    """Paper-faithful: gender un-stratified (AUC); age & BMI within each sex (Pearson).
    Returns mean±std across `n_seeds` nested-CV repeats."""
    rows = [dict(target="Gender", sex="all", metric="auc",
                 **dict(zip(("mean", "std"), _nested(emb_df, "gender_male", "clas", n_seeds)[:2])))]
    for sv, sn in ((1.0, "male"), (0.0, "female")):
        sub = emb_df[emb_df["gender_male"] == sv]
        for ycol, name in (("age", "Age"), ("bmi", "BMI")):
            mu, sd, _ = _nested(sub, ycol, "reg", n_seeds=n_seeds)
            rows.append(dict(target=name, sex=sn, metric="pearson", mean=mu, std=sd))
    df = pd.DataFrame(rows)
    df["mean"] = df["mean"].round(3); df["std"] = df["std"].round(3)
    return df


def nested_report(emb_df: pd.DataFrame, out_dir: str = "results", prefix: str = "ntds_5task",
                  n_seeds: int = 15):
    """Nested-CV metrics table + sex-stratified plots in one pass. Returns (metrics_df, {name: path})."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    os.makedirs(out_dir, exist_ok=True)
    rows, paths = [], {}

    gm, gs, (gyt, gyp) = _nested(emb_df, "gender_male", "clas", n_seeds=n_seeds)
    rows.append(dict(target="Gender", sex="all", metric="auc", mean=round(gm, 3), std=round(gs, 3)))
    fpr, tpr, _ = roc_curve(gyt.astype(int), gyp)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {gm:.3f} ± {gs:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="0.6"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_aspect("equal")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"Gender · nested CV × {n_seeds} seeds"); ax.legend(loc="lower right")
    p = os.path.join(out_dir, f"{prefix}_gender_roc.png"); fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    paths["gender"] = p

    sex_color = {"male": "#2c7fb8", "female": "#d95f0e"}
    for ycol, name, unit in (("age", "Age", "yr"), ("bmi", "BMI", "kg/m²")):
        fig, ax = plt.subplots(figsize=(5, 5)); lims = []
        for sv, sn in ((1.0, "male"), (0.0, "female")):
            mu, sd, (yt, yp) = _nested(emb_df[emb_df["gender_male"] == sv], ycol, "reg", n_seeds=n_seeds)
            rows.append(dict(target=name, sex=sn, metric="pearson", mean=round(mu, 3), std=round(sd, 3)))
            ax.scatter(yt, yp, s=12, alpha=0.4, color=sex_color[sn], edgecolor="none",
                       label=f"{sn}: r = {mu:.2f} ± {sd:.2f}")
            lims += [yt.min(), yt.max(), yp.min(), yp.max()]
        lo, hi = min(lims), max(lims); ax.plot([lo, hi], [lo, hi], "--", color="0.6")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"True {name} ({unit})"); ax.set_ylabel(f"Predicted {name} ({unit})")
        ax.set_title(f"{name} · sex-stratified · nested CV × {n_seeds}"); ax.legend(loc="upper left", fontsize=8)
        p = os.path.join(out_dir, f"{prefix}_{name.lower()}_scatter.png")
        fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
        paths[name.lower()] = p

    return pd.DataFrame(rows), paths


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Probe ntds gait embeddings + render plots.")
    ap.add_argument("--emb", default="results/ntds_gait_smoke_emb.parquet")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--prefix", default="ntds_gait_smoke")
    a = ap.parse_args()
    df = pd.read_parquet(a.emb)
    print(probe_all(df).to_string(index=False))
    print("\nplots:", make_plots(df, a.out_dir, a.prefix))
