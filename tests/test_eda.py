import numpy as np
import pandas as pd
import pytest
from src.eda import EDA

@pytest.fixture
def fake_df():
    """Génère 3 séries, dont B qui est juste A + du bruit (fortement corrélées)."""
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.default_rng(2)
    a = 100 + rng.standard_normal(120).cumsum()
    b = a + rng.standard_normal(120) * 0.1  # B copie A
    c = 50 + rng.standard_normal(120).cumsum() # C est à part
    return pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)

def test_corr_matrix(fake_df):
    eda = EDA(fake_df)
    res = eda.correlation_matrix(max_lag=3)
    
    # 3 séries => matrice 3x3
    assert res["corr"].shape == (3, 3)
    # A et B doivent être ultra corrélées
    assert abs(res["corr"].loc["A", "B"]) > 0.5 

def test_dtw_matrix_is_logic(fake_df):
    eda = EDA(fake_df)
    D = eda.dtw_distance_matrix()
    
    # Matrice symétrique
    np.testing.assert_array_almost_equal(D.values, D.values.T)
    # Diago à zéro (distance avec soi-même)
    np.testing.assert_array_almost_equal(np.diag(D.values), np.zeros(3))
    
    # A et B sont proches, A et C non
    assert D.loc["A", "B"] < D.loc["A", "C"]

def test_seasonality(fake_df):
    eda = EDA(fake_df)
    out = eda.seasonality_tracker(period=7) # Période 7 jours
    
    assert set(out.keys()) == {"A", "B", "C"}
    # On doit retrouver les 3 composantes de base
    assert "trend" in out["A"]
    assert "seasonal" in out["A"]
