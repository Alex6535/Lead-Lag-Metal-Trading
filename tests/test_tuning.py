import numpy as np
import pandas as pd
import pytest

from src.tuning import ModelTuningValidation


@pytest.fixture
def shifted_df():
    n, k = 250, 5
    t = np.arange(n)
    a = np.sin(2 * np.pi * t / 30) + 0.05 * np.random.default_rng(0).standard_normal(n)
    b = np.sin(2 * np.pi * (t - k) / 30) + 0.05 * np.random.default_rng(1).standard_normal(n)
    return pd.DataFrame({"A": a, "B": b})


def test_metrics_on_perfect_forecast():
    s = pd.Series(np.arange(50, dtype=float))
    m = ModelTuningValidation._metrics(s, s.copy())
    assert m["mse"] == 0
    assert m["wass"] == 0


def test_validate_returns_metrics(shifted_df):
    mt = ModelTuningValidation(shifted_df, leader="A", follower="B", n_splits=3)
    out = mt.validate_model(radius=15)
    for k in ("mse", "mae", "rmse", "wass", "lag", "radius"):
        assert k in out
    assert out["lag"] >= 0


def test_tune_runs(shifted_df):
    mt = ModelTuningValidation(shifted_df, leader="A", follower="B", n_splits=3)
    res = mt.tune_model(n_trials=4, radius_range=(5, 15))
    assert "sakoe_chiba_radius" in res["best_params"]
    assert np.isfinite(res["best_value"])
