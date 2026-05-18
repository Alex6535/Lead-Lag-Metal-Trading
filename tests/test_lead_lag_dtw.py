import numpy as np
import pandas as pd
import pytest
from src.lead_lag_dtw import LeadLagDTW

@pytest.fixture
def fake_shifted_pair():
    """Génère 2 sinus. La série A lead la B de k time steps."""
    n, k = 300, 7 
    t = np.arange(n)
    
    rng0, rng1 = np.random.default_rng(0), np.random.default_rng(1)
    a = np.sin(2 * np.pi * t / 40) + 0.05 * rng0.standard_normal(n)
    b = np.sin(2 * np.pi * (t - k) / 40) + 0.05 * rng1.standard_normal(n)
    
    return pd.DataFrame({"A": a, "B": b}), k

def test_if_dtw_finds_the_lag(fake_shifted_pair):
    df, true_lag = fake_shifted_pair
    # On met un radius assez large pour qu'il puisse trouver k=7
    ll = LeadLagDTW(df, sakoe_chiba_radius=20)
    res = ll.identify_lead_lag()
    
    L = res["lag"]
    # A lead B => L[A,B] doit être positif
    assert L.loc["A", "B"] > 0
    # Vérifie qu'on est pas trop loin du vrai lag (DTW c'est pas parfait à l'unité près)
    assert abs(L.loc["A", "B"] - true_lag) <= 3
    # L'anti symétrie est respectée L[A,B] = -L[B,A]
    np.testing.assert_almost_equal(L.loc["A", "B"], -L.loc["B", "A"])

def test_forecast_returns_good_direction(fake_shifted_pair):
    """Teste si prédire en shiftant le leader ça donne une bonne accuracy."""
    df, _ = fake_shifted_pair
    ll = LeadLagDTW(df, sakoe_chiba_radius=20)
    ll.identify_lead_lag()
    
    fc = ll.forecast("A", "B")
    metrics = ll.validate(fc, df["B"])
    
    # On doit avoir une bonne direction de base vu que ce sont 2 sinus parfaits
    assert metrics["dir_acc"] > 0.6
