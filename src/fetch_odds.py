"""Download bookmaker odds for the held-out seasons from football-data.co.uk.

  python src/fetch_odds.py                       # 2023-24, 2024-25, 2025-26
  python src/fetch_odds.py --seasons 2526
If a download is blocked, save https://www.football-data.co.uk/mmz4281/<season>/E0.csv from your browser
as data/odds/E0-<season>.csv and run the script again: it will use the local file.
"""
import argparse
import io
import os
import sys
import urllib.request

import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "odds")
ODDS_COLS = ["B365H", "B365D", "B365A", "B365CH", "B365CD", "B365CA",
             "PSH", "PSD", "PSA", "PSCH", "PSCD", "PSCA",
             "AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2324", "2425", "2526"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for s in a.seasons:
        local = os.path.join(OUT, f"E0-{s}.csv")
        if os.path.exists(local):
            text = open(local, encoding="latin-1").read()
        else:
            url = f"https://www.football-data.co.uk/mmz4281/{s}/E0.csv"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                text = urllib.request.urlopen(req, timeout=30).read().decode("latin-1")
            except Exception as e:
                print(f"[{s}] download failed ({e}). Save {url} as {local} and rerun.")
                continue
        d = pd.read_csv(io.StringIO(text)).dropna(subset=["HomeTeam", "AwayTeam"])
        d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d")
        cols = ["Date", "HomeTeam", "AwayTeam"] + [c for c in ODDS_COLS if c in d.columns]
        d[cols].to_csv(os.path.join(OUT, f"season-{s}.csv"), index=False)
        print(f"[{s}] saved {len(d)} matches, odds columns: {cols[3:]}")


if __name__ == "__main__":
    main()
