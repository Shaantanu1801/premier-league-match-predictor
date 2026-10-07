"""Streamlit front end for the PL match outcome model.  Run: streamlit run app/streamlit_app.py"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
from predict import fixture_row, load_state, score  # noqa: E402

LABELS = ["Home win", "Draw", "Away win"]
st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="wide")


@st.cache_resource(show_spinner="Loading model and team form...")
def load():
    bundle = joblib.load("models/match_model.joblib")
    raw, feats, teams = load_state()
    return bundle, raw, teams


bundle, raw, teams = load()
last_date = raw.Date.max()
current = sorted(set(raw[raw.season == raw.season.max()].HomeTeam))


def predict_rows(rows):
    X = pd.DataFrame([fixture_row(teams, raw, h, a, d) for h, a, d in rows])
    return score(bundle, X)


def prob_bar(p):
    df = pd.DataFrame({"Outcome": LABELS, "Probability": p}).set_index("Outcome")
    st.bar_chart(df, horizontal=True)


st.title("⚽ Premier League Match Outcome Predictor")
st.caption(f"Elo + rolling-form features, logistic regression / XGBoost ensemble. "
           f"Team form is as of the last match in the data: **{last_date.date()}**.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Predict a match", "Fixture list & export", "Matchup grid", "Team ratings", "Model performance"])

# ---------------- single match ----------------
with tab1:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", current, index=current.index("Arsenal") if "Arsenal" in current else 0)
    away_opts = [t for t in current if t != home]
    away = c2.selectbox("Away team", away_opts)
    date = pd.Timestamp(st.date_input("Match date", last_date + pd.Timedelta(days=7)))
    p = predict_rows([(home, away, date)])[0]
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home} win", f"{p[0]*100:.0f}%")
    m2.metric("Draw", f"{p[1]*100:.0f}%")
    m3.metric(f"{away} win", f"{p[2]*100:.0f}%")
    prob_bar(p)
    with st.expander("Why? Key inputs for this match"):
        r = fixture_row(teams, raw, home, away, date)
        info = pd.DataFrame({
            "": ["Elo rating", "Points, last 5 games", "Goals scored, last 5", "Goals conceded, last 5",
                 "Shots on target, last 5", "Days since last match"],
            home: [r["h_elo"], r["h_pts5"], r["h_gf5"], r["h_ga5"], r["h_sot5"], r["h_rest"]],
            away: [r["a_elo"], r["a_pts5"], r["a_gf5"], r["a_ga5"], r["a_sot5"], r["a_rest"]],
        }).set_index("")
        st.dataframe(info.round(2))
    st.info("Draws are the hardest outcome to forecast; treat the draw probability as a base-rate-like estimate, "
            "not a prediction that a draw will happen. No injuries, line-ups or transfers are modelled.")

# ---------------- fixtures + export ----------------
with tab2:
    st.write("Upload a CSV with columns `Date, HomeTeam, AwayTeam` (team names as in the data, e.g. `Man City`).")
    up = st.file_uploader("Fixtures CSV", type="csv")
    sample = pd.DataFrame({"Date": [str((last_date + pd.Timedelta(days=7)).date())] * 2,
                           "HomeTeam": ["Arsenal", "Liverpool"], "AwayTeam": ["Chelsea", "Man City"]})
    st.download_button("Download sample CSV", sample.to_csv(index=False), "fixtures_sample.csv")
    if up is not None:
        fx = pd.read_csv(up, parse_dates=["Date"])
        bad = sorted((set(fx.HomeTeam) | set(fx.AwayTeam)) - set(teams))
        if bad:
            st.error(f"Unknown team names: {bad}")
        else:
            pr = predict_rows(list(zip(fx.HomeTeam, fx.AwayTeam, fx.Date)))
            out = fx.copy()
            out[["p_home", "p_draw", "p_away"]] = pr.round(3)
            out["most_likely"] = np.array(LABELS)[pr.argmax(1)]
            st.dataframe(out, width="stretch")
            st.download_button("Download CSV", out.to_csv(index=False), "predictions.csv")
            js = out.assign(Date=out.Date.astype(str)).to_json(orient="records", indent=2)
            st.download_button("Download JSON (for Supabase)", js, "predictions.json")
            has_sb = "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets
            if has_sb:
                table = st.text_input("Supabase table", "model_predictions")
                if st.button("Push to Supabase"):
                    import requests
                    r = requests.post(f"{st.secrets['SUPABASE_URL']}/rest/v1/{table}", data=js,
                                      headers={"apikey": st.secrets["SUPABASE_KEY"],
                                               "Authorization": f"Bearer {st.secrets['SUPABASE_KEY']}",
                                               "Content-Type": "application/json"}, timeout=30)
                    st.success("Pushed") if r.ok else st.error(f"{r.status_code}: {r.text[:200]}")
            else:
                st.caption("To enable one-click push, add `SUPABASE_URL` and `SUPABASE_KEY` to Streamlit secrets "
                           "and create a table with columns Date, HomeTeam, AwayTeam, p_home, p_draw, p_away, most_likely.")

# ---------------- matchup grid ----------------
with tab3:
    st.write("Probability that the **row team wins at home** against the column team.")
    d0 = last_date + pd.Timedelta(days=7)
    pairs = [(h, a) for h in current for a in current if h != a]
    pr = predict_rows([(h, a, d0) for h, a in pairs])
    grid = pd.DataFrame(np.nan, index=current, columns=current)
    for (h, a), p in zip(pairs, pr):
        grid.loc[h, a] = p[0] * 100
    st.dataframe(grid.style.background_gradient(cmap="Blues", axis=None).format("{:.0f}", na_rep="–"),
                 width="stretch", height=600)

# ---------------- ratings ----------------
with tab4:
    r = pd.DataFrame({"Team": current, "Elo": [teams[t].elo for t in current]}).sort_values("Elo", ascending=False)
    st.bar_chart(r.set_index("Team"), horizontal=True, height=520)
    st.dataframe(r.round(0).reset_index(drop=True), width="stretch")

# ---------------- performance ----------------
with tab5:
    diag = json.load(open("results/diagnostics.json"))
    st.subheader("Held-out test seasons: " + ", ".join(
        f"20{s[:2]}-{s[2:]}" for s in diag["test_seasons"]) + f" ({diag['n_test_matches']} matches)")
    st.dataframe(pd.read_csv("results/metrics_test.csv"), width="stretch")
    st.markdown(
        "- Log loss / Brier: lower is better. ECE: calibration error, lower is better.\n"
        "- Gains over a plain Elo model are small; XGBoost did not beat logistic regression.\n"
        "- The model never makes a draw its top pick; no bookmaker benchmark is included.")
    cols = st.columns(2)
    for i, f in enumerate(["calibration", "model_comparison", "confusion_matrix", "feature_importance"]):
        path = f"results/figures/{f}.png"
        if os.path.exists(path):
            cols[i % 2].image(path, caption=f.replace("_", " ").title())
