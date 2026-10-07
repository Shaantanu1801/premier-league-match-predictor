"""Add / refresh a season of results in data/raw (default: 2026-27).

Source: football-data.co.uk (same data the open dataset is built from).

  python src/update_data.py                       # download 2026-27 and save as data/raw/season-2627.csv
  python src/update_data.py --file E0.csv         # use a file you downloaded by hand instead
  python src/update_data.py --season 2526         # another season (code = start yy + end yy)

Run it again any time to pick up new matchdays, then retrain:  python src/train.py
"""
import argparse
import io
import os
import sys
import urllib.request

import pandas as pd

KEEP = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR", "Referee",
        "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"]
NEED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2627")
    ap.add_argument("--file", help="local CSV downloaded from football-data.co.uk")
    ap.add_argument("--out-dir", default=RAW)
    a = ap.parse_args()

    if a.file:
        text = open(a.file, encoding="latin-1").read()
    else:
        url = f"https://www.football-data.co.uk/mmz4281/{a.season}/E0.csv"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            text = urllib.request.urlopen(req, timeout=30).read().decode("latin-1")
        except Exception as e:
            sys.exit(f"Download failed ({e}).\nOpen {url} in your browser, save it, then run:\n"
                     f"  python src/update_data.py --file <saved file> --season {a.season}")

    d = pd.read_csv(io.StringIO(text)).dropna(how="all")
    missing = [c for c in NEED if c not in d.columns]
    if missing:
        sys.exit(f"Unexpected file format, missing columns: {missing}")
    d = d.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])   # only played matches
    d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d")
    for c in ["FTHG", "FTAG"]:
        d[c] = d[c].astype(int)
    for c in KEEP:                                   # keep schema identical to older seasons
        if c not in d.columns:
            d[c] = pd.NA
    d = d[KEEP].sort_values("Date", kind="stable")

    # sanity: warn about team names the model has not seen
    known = set()
    for f in os.listdir(a.out_dir):
        if f.startswith("season-") and f != f"season-{a.season}.csv":
            t = pd.read_csv(os.path.join(a.out_dir, f), usecols=["HomeTeam"])
            known |= set(t.HomeTeam)
    new = sorted((set(d.HomeTeam) | set(d.AwayTeam)) - known)
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"season-{a.season}.csv")
    d.to_csv(out, index=False)
    print(f"saved {len(d)} played matches -> {out} (latest match: {d.Date.max()})")
    if new:
        print("teams new to the model (expected for promoted clubs, otherwise check spelling):", new)
    if d[["HS", "HST", "HC"]].isna().any().any():
        print("note: some matches have no shots/corners data; those features will be blank for them.")


if __name__ == "__main__":
    main()
