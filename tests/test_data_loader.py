import numpy as np
import pandas as pd
import pytest
from src.data_loader import DataLoader

def fake_data_loader():
    """Génère un faux loader avec des data random pour pas taper l'API Yahoo pdant les tests"""
    idx = pd.date_range("2024-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    
    a = pd.Series(100 + rng.standard_normal(50).cumsum(), index=idx, name="A")
    b = pd.Series(50 + rng.standard_normal(50).cumsum(), index=idx, name="B")
    
    # On met quelques NaNs pour tester l'imputation
    a.iloc[5:8] = np.nan
    b.iloc[20] = np.nan
    
    dl = DataLoader(tickers={"A": "A", "B": "B"}, start_date="2024-01-01", end_date="2024-03-01")
    dl.raw_data = {"A": a, "B": b} # Trick: on set directement raw_data
    return dl

def test_merge_data():
    dl = fake_data_loader()
    merged = dl.merge_data()
    # On check que ça a bien concaténé les 2 colonnes
    assert list(merged.columns) == ["A", "B"]
    assert merged.shape == (50, 2)

def test_impute_data():
    dl = fake_data_loader()
    dl.merge_data()
    out = dl.impute_data()
    # Il ne doit rester AUCUN NaN après le ffill/bfill
    assert not out.isna().any().any()

def test_missing_test_structure():
    dl = fake_data_loader()
    dl.merge_data()
    res = dl.missing_test()
    # On vérifie juste qu'on a bien nos clés de retour
    assert "per_column" in res
    assert "mcar_plausible" in res

def test_merge_empty_crashes():
    # Si on a pas fetch, ça doit crash
    dl = DataLoader(tickers={}, start_date="2024-01-01", end_date="2024-02-01")
    with pytest.raises(ValueError):
        dl.merge_data()
