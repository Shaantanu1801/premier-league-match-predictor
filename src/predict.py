"""Predict match outcomes with the saved model.

Examples
--------
python src/predict.py --home Arsenal --away Chelsea
python src/predict.py --fixtures fixtures.csv --out predictions.csv   # cols: Date,HomeTeam,AwayTeam
python src/predict.py --ratings                                        # current Elo table

Team form is taken from the end of the data in data/raw. Fixtures are scored as if
played right after the last match in the data, so refresh the raw CSVs first.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from features import build_features, load_raw, FEATURES, _team_features, _expected


def load_state():
    raw = load_raw()
    feats, teams = build_features(raw)
    return raw, feats, teams


def fixture_row(teams, hist, home, away, date):
    for t in (home, away):
        if t not in teams:
            raise SystemExit(f"Unknown team '{t}'. Use names like: {sorted(teams)[:8]} ...")
    sh, sa = teams[home], teams[away]
    f = {}
    f.update(_team_features("h", sh, date, sh.home5))
    f.update(_team_features("a", sa, date, sa.away5))
    # "promoted" = not in last season's table; if the latest season in the data is still in
    # progress (<380 matches) the reference season is the one before it (matches train-time logic)
    seasons = sorted(hist.season.unique())
    n_latest = (hist.season == seasons[-1]).sum()
    ref = seasons[-2] if (n_latest < 380 and len(seasons) > 1) else seasons[-1]
    last = hist[hist.season == ref]
    in_season = set(last.HomeTeam) | set(last.AwayTeam)
    f["h_promoted"], f["a_promoted"] = int(home not in in_season), int(away not in in_season)
    prior = hist[((hist.HomeTeam == home) & (hist.AwayTeam == away)) |
                 ((hist.HomeTeam == away) & (hist.AwayTeam == home))].tail(5)
    pts = [3 if (r.HomeTeam == home and r.FTHG > r.FTAG) or (r.HomeTeam == away and r.FTAG > r.FTHG)
           else 1 if r.FTHG == r.FTAG else 0 for r in prior.itertuples()]
    f["h2h_home_ppg"], f["h2h_n"] = (float(np.mean(pts)) if pts else np.nan), len(pts)
    f["elo_diff"] = f["h_elo"] - f["a_elo"]
    f["exp_home"] = _expected(f["h_elo"], f["a_elo"])
    for k in ["pts5", "pts10", "gf5", "ga5", "sot5", "sot_a5", "sot10"]:
        f[f"{k}_diff"] = f[f"h_{k}"] - f[f"a_{k}"]
    f["season_ppg_diff"] = f["h_season_ppg"] - f["a_season_ppg"]
    f["rest_diff"] = f["h_rest"] - f["a_rest"]
    return f


def score(bundle, X):
    X = X[FEATURES]
    p_lr = bundle["lr"].predict_proba(X)
    p_xgb = bundle["xgb"].predict_proba(X)
    return {"Logistic regression": p_lr, "XGBoost": p_xgb,
            "Ensemble (LR + XGB)": (p_lr + p_xgb) / 2}[bundle["chosen"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home"); ap.add_argument("--away")
    ap.add_argument("--fixtures"); ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--ratings", action="store_true")
    a = ap.parse_args()

    bundle = joblib.load("models/match_model.joblib")
    raw, feats, teams = load_state()
    last_date = raw.Date.max()
    if a.ratings:
        cur = set(raw[raw.season == raw.season.max()].HomeTeam)
        r = sorted(((teams[t].elo, t) for t in cur), reverse=True)
        print(pd.DataFrame(r, columns=["elo", "team"]).round(0).to_string(index=False)); return

    if a.fixtures:
        fx = pd.read_csv(a.fixtures, parse_dates=["Date"])
    elif a.home and a.away:
        fx = pd.DataFrame([{"Date": last_date + pd.Timedelta(days=7), "HomeTeam": a.home, "AwayTeam": a.away}])
    else:
        raise SystemExit("give --home/--away, --fixtures, or --ratings")

    X = pd.DataFrame([fixture_row(teams, raw, r.HomeTeam, r.AwayTeam, r.Date) for r in fx.itertuples()])
    p = score(bundle, X)
    out = fx.copy()
    out[["p_home", "p_draw", "p_away"]] = p.round(3)
    out["most_likely"] = np.array(["Home win", "Draw", "Away win"])[p.argmax(1)]
    print(out.to_string(index=False))
    out.to_csv(a.out, index=False)
    out.assign(Date=out.Date.astype(str)).to_json(a.out.replace(".csv", ".json"), orient="records", indent=2)
    print(f"\nsaved -> {a.out} (+ .json for Supabase upload). Data ends {last_date.date()}.")


if __name__ == "__main__":
    main()
