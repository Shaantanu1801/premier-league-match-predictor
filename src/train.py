"""Walk-forward training + evaluation of PL match outcome models.

Design
------
* Tuning only ever sees VAL seasons (2020-21..2022-23), walk-forward.
* Final numbers are on TEST seasons (2023-24..2025-26), walk-forward, never tuned on.
* For each season S the model is trained on all seasons < S (first season dropped: Elo burn-in).
* Probability calibration (vector scaling) is fit on the VAL out-of-sample predictions only.
"""
import itertools
import json
import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from calibration import VectorCalibrator
from features import FEATURES, build_features, load_raw

warnings.filterwarnings("ignore")
SEED = 42
VAL = ["2021", "2122", "2223"]
TEST = ["2324", "2425", "2526"]
CLASSES = ["Home win", "Draw", "Away win"]
os.makedirs("results/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)


def brier(y, p):
    oh = np.eye(3)[y]
    return float(np.mean(np.sum((p - oh) ** 2, axis=1)))


def ece(y, p, bins=10):
    """Multiclass expected calibration error (one-vs-rest averaged)."""
    out = []
    for c in range(3):
        pc, yc = p[:, c], (y == c).astype(float)
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(pc, edges) - 1, 0, bins - 1)
        e = sum(abs(yc[idx == b].mean() - pc[idx == b].mean()) * (idx == b).mean()
                for b in range(bins) if (idx == b).any())
        out.append(e)
    return float(np.mean(out))


def metrics(y, p):
    return dict(log_loss=log_loss(y, p, labels=[0, 1, 2]), brier=brier(y, p),
                accuracy=accuracy_score(y, p.argmax(1)), ece=ece(y, p))


# ---------------- models ----------------
def m_prior(Xtr, ytr, Xte, **_):
    return np.tile(np.bincount(ytr, minlength=3) / len(ytr), (len(Xte), 1))


def m_home(Xtr, ytr, Xte, **_):  # "always pick home": hard prediction, smoothed one-hot
    p = np.tile([0.8, 0.1, 0.1], (len(Xte), 1))
    return p


def make_lr(cols, C=0.1):
    def f(Xtr, ytr, Xte, **_):
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(Xtr[cols], ytr)
        return pipe.predict_proba(Xte[cols])
    return f


def make_xgb(params):
    def f(Xtr, ytr, Xte, return_model=False, **_):
        mdl = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
                            subsample=0.8, colsample_bytree=0.7, random_state=SEED,
                            n_jobs=4, eval_metric="mlogloss", **params)
        mdl.fit(Xtr[FEATURES], ytr)
        p = mdl.predict_proba(Xte[FEATURES])
        return (p, mdl) if return_model else p
    return f


def walk_forward(df, fn, seasons):
    parts = []
    for s in seasons:
        tr = df[(df.season < s) & (df.season != "0001")]
        te = df[df.season == s]
        p = fn(tr, tr["y"].values, te)
        parts.append(pd.DataFrame(p, index=te.index, columns=[0, 1, 2]))
    return pd.concat(parts).sort_index()


def main():
    raw = load_raw()
    df, _ = build_features(raw)
    df.to_csv("data/processed/features.csv", index=False)
    y = df["y"].values

    # ---- hyper-parameter tuning on VAL seasons only
    lr_grid = {C: walk_forward(df, make_lr(FEATURES, C), VAL) for C in [0.003, 0.01, 0.03, 0.1, 1.0]}
    yv = df.loc[df.season.isin(VAL), "y"].values
    best_C = min(lr_grid, key=lambda C: log_loss(yv, lr_grid[C].values))
    xgb_results = []
    grid = itertools.product([2, 3], [0.02], [100, 200, 350], [10, 25], [1.0, 10.0])
    for d, lr, n, mcw, lam in grid:
        params = dict(max_depth=d, learning_rate=lr, n_estimators=n, min_child_weight=mcw, reg_lambda=lam)
        pv = walk_forward(df, make_xgb(params), VAL)
        xgb_results.append((log_loss(yv, pv.values), params))
    xgb_results.sort(key=lambda t: t[0])
    best_params = xgb_results[0][1]
    print("best LR C:", best_C, "| best XGB:", best_params, "val LL=%.4f" % xgb_results[0][0])

    # ---- calibrator fit on VAL out-of-sample XGB predictions
    pv_best = walk_forward(df, make_xgb(best_params), VAL).values
    calib = VectorCalibrator().fit(pv_best, yv)

    # ---- TEST evaluation
    models = {
        "Class prior": m_prior,
        "Always home win": m_home,
        "Elo only (LR)": make_lr(["elo_diff"], 1.0),
        "Logistic regression": make_lr(FEATURES, best_C),
        "XGBoost": make_xgb(best_params),
    }
    yt = df.loc[df.season.isin(TEST), "y"].values
    preds, rows = {}, []
    for name, fn in models.items():
        p = walk_forward(df, fn, TEST).values
        preds[name] = p
        rows.append({"model": name, **metrics(yt, p)})
    preds["XGBoost + calibration"] = calib.predict(preds["XGBoost"])
    rows.append({"model": "XGBoost + calibration", **metrics(yt, preds["XGBoost + calibration"])})
    preds["Ensemble (LR + XGB)"] = (preds["Logistic regression"] + preds["XGBoost"]) / 2
    rows.append({"model": "Ensemble (LR + XGB)", **metrics(yt, preds["Ensemble (LR + XGB)"])})

    # model of record is chosen on VALIDATION log loss only (never on test)
    pv_lr = lr_grid[best_C].values
    val_ll = {"Logistic regression": log_loss(yv, pv_lr),
              "XGBoost": log_loss(yv, pv_best),
              "Ensemble (LR + XGB)": log_loss(yv, (pv_lr + pv_best) / 2)}
    chosen = min(val_ll, key=val_ll.get)
    print("validation log loss:", {k: round(v, 4) for k, v in val_ll.items()}, "-> chosen:", chosen)
    res = pd.DataFrame(rows).round(4)
    res.to_csv("results/metrics_test.csv", index=False)
    print(res.to_string(index=False))

    # per-season
    per = []
    te = df[df.season.isin(TEST)].reset_index(drop=True)
    for name, p in preds.items():
        for s in TEST:
            msk = (te.season == s).values
            per.append({"model": name, "season": s, **metrics(yt[msk], p[msk])})
    pd.DataFrame(per).round(4).to_csv("results/metrics_by_season.csv", index=False)

    # ---- save test predictions
    out = te[["Date", "season", "HomeTeam", "AwayTeam", "FTR"]].copy()
    best = preds[chosen]
    out[["p_home", "p_draw", "p_away"]] = best.round(4)
    out["pred"] = np.array(["H", "D", "A"])[best.argmax(1)]
    out.to_csv("results/test_predictions.csv", index=False)

    # ---- extra diagnostics
    diag = {"best_C": best_C, "best_xgb_params": best_params,
            "validation_log_loss": {k: float(v) for k, v in val_ll.items()}, "chosen_model": chosen,
            "n_test_matches": int(len(yt)),
            "draws_predicted_pct": float((best.argmax(1) == 1).mean() * 100),
            "draw_actual_pct": float((yt == 1).mean() * 100),
            "class_recall": {c: float(((best.argmax(1) == i) & (yt == i)).sum() / (yt == i).sum())
                             for i, c in enumerate(CLASSES)},
            "train_seasons": "2001-02 .. (season-1) walk-forward", "test_seasons": TEST}
    json.dump(diag, open("results/diagnostics.json", "w"), indent=2)
    print(json.dumps(diag, indent=2))

    # ---- figures
    plot_comparison(res)
    plot_calibration(yt, preds["XGBoost"], preds[chosen], chosen)
    plot_confusion(yt, best)
    plot_prob_dist(yt, best)

    # ---- final models on all data for deployment
    all_train = df[df.season != "0001"]
    lr_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(C=best_C, max_iter=2000)).fit(all_train[FEATURES], all_train["y"])
    _, mdl = make_xgb(best_params)(all_train, all_train["y"].values, all_train.tail(5), return_model=True)
    joblib.dump({"lr": lr_pipe, "xgb": mdl, "calibrator": calib, "features": FEATURES,
                 "chosen": chosen, "params": best_params}, "models/match_model.joblib")
    plot_importance(mdl, all_train)


# ---------------- plots ----------------
def plot_comparison(res):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = res[~res.model.eq("Always home win")]
    ax[0].barh(r.model, r.log_loss, color="#3b6ea5")
    ax[0].set_xlim(r.log_loss.min() - 0.02, r.log_loss.max() + 0.01)
    ax[0].invert_yaxis(); ax[0].set_title("Log loss (lower is better)")
    for i, v in enumerate(r.log_loss): ax[0].text(v, i, f" {v:.4f}", va="center", fontsize=8)
    ax[1].barh(res.model, res.accuracy * 100, color="#6aa84f")
    ax[1].invert_yaxis(); ax[1].set_title("Accuracy % (argmax)")
    for i, v in enumerate(res.accuracy * 100): ax[1].text(v, i, f" {v:.1f}", va="center", fontsize=8)
    plt.tight_layout(); plt.savefig("results/figures/model_comparison.png", dpi=150); plt.close()


def plot_calibration(y, p_raw, p_cal, label="chosen model"):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for c, name in enumerate(CLASSES):
        for p, lab, col in [(p_raw, "XGBoost raw", "#cc4125"), (p_cal, label + " (chosen)", "#3b6ea5")]:
            q = pd.qcut(p[:, c], 8, duplicates="drop")
            g = pd.DataFrame({"p": p[:, c], "y": (y == c).astype(int), "q": q}).groupby("q", observed=True).mean()
            ax[c].plot(g.p, g.y, "o-", label=lab, color=col)
        ax[c].plot([0, 1], [0, 1], "k--", lw=1)
        ax[c].set_title(name); ax[c].set_xlabel("Predicted probability"); ax[c].set_ylabel("Observed frequency")
        lim = max(p_raw[:, c].max(), p_cal[:, c].max()) + 0.05
        ax[c].set_xlim(0, lim); ax[c].set_ylim(0, lim)
    ax[0].legend(fontsize=8)
    plt.suptitle("Reliability (test seasons, 8 equal-count bins)")
    plt.tight_layout(); plt.savefig("results/figures/calibration.png", dpi=150); plt.close()


def plot_confusion(y, p):
    cm = pd.crosstab(pd.Series(y, name="Actual"), pd.Series(p.argmax(1), name="Predicted"))
    cm = cm.reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0)
    cmn = cm.div(cm.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cmn.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm.values[i, j]}\n({cmn.values[i, j]*100:.0f}%)", ha="center", va="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(["H", "D", "A"]); ax.set_yticks(range(3)); ax.set_yticklabels(["H", "D", "A"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (row-normalised)")
    plt.tight_layout(); plt.savefig("results/figures/confusion_matrix.png", dpi=150); plt.close()


def plot_prob_dist(y, p):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(p[y == 1, 1], bins=20, alpha=.6, label="actual draws", density=True)
    ax.hist(p[y != 1, 1], bins=20, alpha=.6, label="non-draws", density=True)
    ax.set_xlabel("Predicted draw probability"); ax.legend(); ax.set_title("Draws are barely separable")
    plt.tight_layout(); plt.savefig("results/figures/draw_probability.png", dpi=150); plt.close()


def plot_importance(mdl, train):
    imp, title = None, "XGBoost gain importance"
    try:
        import shap
        sample = train.sample(1500, random_state=SEED)[FEATURES]
        sv = np.abs(np.array(shap.TreeExplainer(mdl).shap_values(sample)))
        axes = [i for i, n in enumerate(sv.shape) if n == len(FEATURES)]
        other = tuple(i for i in range(sv.ndim) if i != axes[0])
        imp = pd.Series(sv.mean(axis=other), index=FEATURES)
        title = "Mean |SHAP| across classes (XGBoost)"
    except Exception as e:
        print("SHAP unavailable, using gain:", e)
    if imp is None:
        imp = pd.Series(mdl.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
    imp = imp.sort_values().tail(15)
    imp.to_csv("results/feature_importance.csv", header=["importance"])
    fig, ax = plt.subplots(figsize=(6, 5)); ax.barh(imp.index, imp.values, color="#3b6ea5")
    ax.set_title(title); plt.tight_layout(); plt.savefig("results/figures/feature_importance.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
