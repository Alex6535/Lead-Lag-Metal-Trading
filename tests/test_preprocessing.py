import numpy as np
import pandas as pd
import pytest
from src.preprocessing import Preprocessing

@pytest.fixture
def fake_df():
    """Génère 200 jours de prix bidons pour tester les filtres"""
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.default_rng(1)
    a = 100 + rng.standard_normal(200).cumsum()
    b = 50 + rng.standard_normal(200).cumsum()
    return pd.DataFrame({"A": a, "B": b}, index=idx)

def test_transform_data(fake_df):
    pp = Preprocessing(fake_df)
    out = pp.transform_data(ma_window=10)
    
    # 1 log-return perdu (diff) => 199 lignes
    assert out["log_returns"].shape == (199, 2)
    assert out["moving_average"].shape == fake_df.shape
    assert out["scaled"].shape == fake_df.shape
    
    # Le RobustScaler doit nous centrer autour de 0 environ
    assert abs(out["scaled"].median().mean()) < 0.5

def test_all_filters_dont_crash(fake_df):
    """Vérifie juste que les filtres tournent et sortent la même shape"""
    pp = Preprocessing(fake_df)
    filters = [
        pp.apply_kalman_filter,
        pp.apply_butterworth_filter,
        pp.apply_savgol_filter,
        pp.apply_moving_average,
        pp.apply_ta_lib_filter,
    ]
    
    for f in filters:
        res = f("A")
        assert len(res) == 200, f"Le filtre {f.__name__} a drop des lignes"
        assert res.notna().sum() > 100 # On vérifie qu'on n'a pas que des NaNs

def test_apply_all_helper(fake_df):
    pp = Preprocessing(fake_df)
    out = pp.apply_all_filters("A")
    # Raw + 5 filtres = 6 colonnes
    assert out.shape[1] == 6
