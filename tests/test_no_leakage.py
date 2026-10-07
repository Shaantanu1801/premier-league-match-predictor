"""Leakage check: scrambling every result after a cutoff must not change any earlier feature row."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from features import build_features, load_raw, FEATURES


def test_future_results_do_not_leak():
    raw = load_raw()
    cut = raw.index[raw.Date >= "2022-01-01"][0]
    a, _ = build_features(raw)
    b_raw = raw.copy()
    rng = np.random.default_rng(0)
    for c in ["FTHG", "FTAG", "HS", "AS", "HST", "AST", "HC", "AC"]:
        b_raw.loc[cut:, c] = rng.integers(0, 6, len(b_raw) - cut)
    b, _ = build_features(b_raw)
    pre_a, pre_b = a.loc[: cut - 1, FEATURES], b.loc[: cut - 1, FEATURES]
    assert np.allclose(pre_a.fillna(-999).values, pre_b.fillna(-999).values)
    # and the first affected match itself must still only use pre-match info
    assert np.allclose(a.loc[cut, FEATURES].fillna(-999).values.astype(float),
                       b.loc[cut, FEATURES].fillna(-999).values.astype(float))


if __name__ == "__main__":
    test_future_results_do_not_leak(); print("PASS: no leakage")
