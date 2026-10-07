"""Leakage-free feature engineering for PL match outcome prediction.

Matches are processed strictly in chronological order. Every feature for a match
is computed from the state *before* kickoff; the state is only updated with the
result afterwards.
"""
import glob
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
ELO_START, ELO_NEWCOMER, ELO_HOME_ADV, ELO_K, ELO_CARRY = 1500.0, 1450.0, 65.0, 20.0, 0.75
STATS = ["pts", "gf", "ga", "sh", "sot", "sh_a", "sot_a", "corn"]


def load_raw(raw_dir=RAW_DIR):
    frames = []
    for f in sorted(glob.glob(os.path.join(raw_dir, "season-*.csv"))):
        d = pd.read_csv(f)
        d["season"] = os.path.basename(f)[7:11]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "season"], kind="stable").reset_index(drop=True)
    return df


def _expected(elo_h, elo_a):
    return 1.0 / (1.0 + 10 ** (-((elo_h + ELO_HOME_ADV) - elo_a) / 400.0))


def _mean(dq, key, min_n=3):
    if len(dq) < min_n:
        return np.nan
    return float(np.mean([m[key] for m in dq]))


class TeamState:
    def __init__(self):
        self.elo = ELO_NEWCOMER
        self.last10 = deque(maxlen=10)
        self.home5 = deque(maxlen=5)
        self.away5 = deque(maxlen=5)
        self.last_date = None
        self.season_pts = 0
        self.season_played = 0
        self.season_gd = 0


def _team_features(prefix, st, date, venue_dq):
    last5 = list(st.last10)[-5:]
    f = {}
    f[f"{prefix}_elo"] = st.elo
    for k in ["pts", "gf", "ga", "sh", "sot", "sh_a", "sot_a", "corn"]:
        f[f"{prefix}_{k}5"] = _mean(last5, k)
    for k in ["pts", "gf", "ga", "sot"]:
        f[f"{prefix}_{k}10"] = _mean(st.last10, k)
    f[f"{prefix}_venue_pts5"] = _mean(venue_dq, "pts")
    f[f"{prefix}_venue_gd5"] = (
        _mean(venue_dq, "gf") - _mean(venue_dq, "ga") if len(venue_dq) >= 3 else np.nan
    )
    f[f"{prefix}_rest"] = (
        min((date - st.last_date).days, 14) if st.last_date is not None else np.nan
    )
    f[f"{prefix}_season_ppg"] = (
        st.season_pts / st.season_played if st.season_played else np.nan
    )
    f[f"{prefix}_season_gdpg"] = (
        st.season_gd / st.season_played if st.season_played else np.nan
    )
    return f


def build_features(df):
    teams = defaultdict(TeamState)
    h2h = defaultdict(lambda: deque(maxlen=5))
    prev_teams, cur_teams, cur_season = set(), set(), None
    rows = []

    for _, m in df.iterrows():
        # --- season rollover: regress Elo towards the mean, reset season tallies
        if m["season"] != cur_season:
            if cur_season is not None:
                prev_teams = set(cur_teams)
                for t, st in teams.items():
                    st.elo = ELO_START + ELO_CARRY * (st.elo - ELO_START)
                    st.season_pts = st.season_played = st.season_gd = 0
            cur_teams, cur_season = set(), m["season"]
        h, a, date = m["HomeTeam"], m["AwayTeam"], m["Date"]
        sh_, sa_ = teams[h], teams[a]
        cur_teams.update([h, a])

        f = {"Date": date, "season": m["season"], "HomeTeam": h, "AwayTeam": a,
             "FTR": m["FTR"], "FTHG": m["FTHG"], "FTAG": m["FTAG"]}
        f.update(_team_features("h", sh_, date, sh_.home5))
        f.update(_team_features("a", sa_, date, sa_.away5))
        f["h_promoted"] = int(bool(prev_teams) and h not in prev_teams)
        f["a_promoted"] = int(bool(prev_teams) and a not in prev_teams)

        key = frozenset([h, a])
        hist = h2h[key]
        if len(hist):
            pts = [3 if (hh == h and hg > ag) or (hh == a and ag > hg)
                   else 1 if hg == ag else 0 for hh, hg, ag in hist]
            f["h2h_home_ppg"], f["h2h_n"] = float(np.mean(pts)), len(hist)
        else:
            f["h2h_home_ppg"], f["h2h_n"] = np.nan, 0

        # differences (what the models mostly use)
        f["elo_diff"] = f["h_elo"] - f["a_elo"]
        f["exp_home"] = _expected(f["h_elo"], f["a_elo"])
        for k in ["pts5", "pts10", "gf5", "ga5", "sot5", "sot_a5", "sot10"]:
            f[f"{k}_diff"] = f[f"h_{k}"] - f[f"a_{k}"]
        f["season_ppg_diff"] = f["h_season_ppg"] - f["a_season_ppg"]
        f["rest_diff"] = f["h_rest"] - f["a_rest"]

        # --- update state with the result (after features are frozen)
        hg, ag = int(m["FTHG"]), int(m["FTAG"])
        res_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        mult = 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8.0
        delta = ELO_K * mult * (res_h - _expected(sh_.elo, sa_.elo))
        sh_.elo += delta
        sa_.elo -= delta
        for st, gf, ga, sh, sot, sha, sota, corn, pts, dq in [
            (sh_, hg, ag, m["HS"], m["HST"], m["AS"], m["AST"], m["HC"],
             3 if hg > ag else 1 if hg == ag else 0, sh_.home5),
            (sa_, ag, hg, m["AS"], m["AST"], m["HS"], m["HST"], m["AC"],
             3 if ag > hg else 1 if hg == ag else 0, sa_.away5),
        ]:
            rec = dict(pts=pts, gf=gf, ga=ga, sh=sh, sot=sot, sh_a=sha, sot_a=sota, corn=corn)
            st.last10.append(rec)
            dq.append(rec)
            st.last_date = date
            st.season_pts += pts
            st.season_played += 1
            st.season_gd += gf - ga
        h2h[key].append((h, hg, ag))
        rows.append(f)

    out = pd.DataFrame(rows)
    out["y"] = out["FTR"].map({"H": 0, "D": 1, "A": 2})
    return out, teams


FEATURES = [
    "elo_diff", "exp_home", "h_elo", "a_elo",
    "pts5_diff", "pts10_diff", "gf5_diff", "ga5_diff", "sot5_diff", "sot_a5_diff", "sot10_diff",
    "h_pts5", "a_pts5", "h_gf5", "a_gf5", "h_ga5", "a_ga5", "h_sot5", "a_sot5",
    "h_sot_a5", "a_sot_a5", "h_corn5", "a_corn5",
    "h_venue_pts5", "a_venue_pts5", "h_venue_gd5", "a_venue_gd5",
    "h_rest", "a_rest", "rest_diff", "h_season_ppg", "a_season_ppg", "season_ppg_diff",
    "h_season_gdpg", "a_season_gdpg", "h_promoted", "a_promoted", "h2h_home_ppg", "h2h_n",
]

if __name__ == "__main__":
    raw = load_raw()
    feats, _ = build_features(raw)
    os.makedirs("data/processed", exist_ok=True)
    feats.to_csv("data/processed/features.csv", index=False)
    print(feats.shape)
