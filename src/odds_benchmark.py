"""Compare the model with bookmaker-implied probabilities on the held-out matches.

  python src/odds_benchmark.py

Bookmaker probabilities = inverse odds, normalised to sum to 1 (removes the margin proportionally).
Closing odds contain late information (line-ups, injuries) that the model does not have, so they are the
hardest benchmark; opening/earlier odds are shown too when available.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOOKS = [  # (label, home, draw, away)
    ("Pinnacle closing", "PSCH", "PSCD", "PSCA"),
    ("Market average closing", "AvgCH", "AvgCD", "AvgCA"),
    ("Bet365 closing", "B365CH", "B365CD", "B365CA"),
    ("Pinnacle", "PSH", "PSD", "PSA"),
    ("Market average", "AvgH", "AvgD", "AvgA"),
    ("Bet365", "B365H", "B365D", "B365A"),
]
Y = {"H": 0, "D": 1, "A": 2}


def ll_vec(y, p):
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))


def brier(y, p):
    return np.sum((p - np.eye(3)[y]) ** 2, axis=1)


def main():
    pred = pd.read_csv("results/test_predictions.csv")
    pred["Date"] = pd.to_datetime(pred["Date"]).dt.strftime("%Y-%m-%d")
    files = sorted(glob.glob("data/odds/season-*.csv"))
    if not files:
        raise SystemExit("No odds files found. Run: python src/fetch_odds.py")
    odds = pd.concat([pd.read_csv(f) for f in files])
    m = pred.merge(odds, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    print(f"Matched {len(m)} of {len(pred)} test matches with odds "
          f"({len(m) / len(pred) * 100:.0f}% coverage)")
    if len(m) < 100:
        raise SystemExit("Too few matches matched. Check team names / dates in data/odds.")
    y = m["FTR"].map(Y).values
    pm = m[["p_home", "p_draw", "p_away"]].values

    rows, books = [], {}
    rows.append(dict(source="Model (ensemble)", n=len(m), log_loss=ll_vec(y, pm).mean(),
                     brier=brier(y, pm).mean(), accuracy=(pm.argmax(1) == y).mean()))
    for label, h, d, a in BOOKS:
        if not all(c in m.columns for c in (h, d, a)):
            continue
        o = m[[h, d, a]].astype(float)
        ok = o.notna().all(axis=1) & (o > 1).all(axis=1)
        if ok.mean() < 0.9:
            continue
        inv = 1.0 / o[ok].values
        pb = inv / inv.sum(1, keepdims=True)
        books[label] = (ok.values, pb)
        # model restricted to the same rows, for a fair like-for-like comparison
        rows.append(dict(source=label, n=int(ok.sum()), log_loss=ll_vec(y[ok], pb).mean(),
                         brier=brier(y[ok], pb).mean(), accuracy=(pb.argmax(1) == y[ok]).mean(),
                         model_log_loss_same_rows=ll_vec(y[ok], pm[ok]).mean(),
                         overround_pct=(inv.sum(1).mean() - 1) * 100))
    res = pd.DataFrame(rows).round(4)
    res.to_csv("results/odds_benchmark.csv", index=False)
    print(res.to_string(index=False))

    # paired bootstrap on per-match log-loss difference (model - bookmaker), best (first) available book
    rng = np.random.default_rng(0)
    label = next(iter(books))
    ok, pb = books[label]
    diff = ll_vec(y[ok], pm[ok]) - ll_vec(y[ok], pb)
    boots = np.array([rng.choice(diff, len(diff)).mean() for _ in range(5000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nModel minus {label}: mean log-loss difference {diff.mean():+.4f} "
          f"(95% CI {lo:+.4f} to {hi:+.4f}); negative = model better")
    verdict = ("the model is statistically indistinguishable from the bookmaker" if lo < 0 < hi else
               "the model beats the bookmaker" if hi < 0 else "the bookmaker is clearly better than the model")
    print("Interpretation:", verdict, "on these matches.")

    # does the model add anything on top of the market? blend p = w*model + (1-w)*book
    ws = np.linspace(0, 1, 21)
    blend = [ll_vec(y[ok], w * pm[ok] + (1 - w) * pb).mean() for w in ws]
    bw = ws[int(np.argmin(blend))]
    print(f"Best blend weight on the model: {bw:.2f} (log loss {min(blend):.4f} vs "
          f"market alone {blend[0]:.4f}). Descriptive only, fitted on these same matches.")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = res.sort_values("log_loss")
    ax[0].barh(r.source, r.log_loss, color=["#cc4125" if s.startswith("Model") else "#3b6ea5" for s in r.source])
    ax[0].invert_yaxis(); ax[0].set_xlim(r.log_loss.min() - 0.02, r.log_loss.max() + 0.01)
    ax[0].set_title("Log loss on the same test matches (lower is better)")
    for i, v in enumerate(r.log_loss): ax[0].text(v, i, f" {v:.4f}", va="center", fontsize=8)
    ax[1].plot(ws, blend, "o-"); ax[1].axvline(bw, ls="--", c="grey")
    ax[1].set_xlabel("Weight on model (0 = market only, 1 = model only)"); ax[1].set_ylabel("Log loss")
    ax[1].set_title(f"Blend with {label}")
    plt.tight_layout(); plt.savefig("results/figures/odds_benchmark.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()
