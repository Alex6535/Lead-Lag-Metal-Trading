import numpy as np
import pandas as pd
import pytest

from src.entropy_graph import EntropyTransferGraph, _jsd


def test_jsd_properties():
    p = np.array([0.5, 0.5])
    q = np.array([0.5, 0.5])
    assert _jsd(p, q) == pytest.approx(0.0, abs=1e-12)
    p2 = np.array([1.0, 0.0])
    q2 = np.array([0.0, 1.0])
    # max JSD with natural log = ln 2
    assert _jsd(p2, q2) == pytest.approx(np.log(2), rel=1e-6)


@pytest.fixture
def D():
    cols = ["A", "B", "C", "D"]
    rng = np.random.default_rng(0)
    M = rng.uniform(0.1, 1.0, size=(4, 4))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 0.0)
    return pd.DataFrame(M, index=cols, columns=cols)


def test_compute_embeddings_mds(D):
    g = EntropyTransferGraph(D)
    out = g.compute_embeddings(method="mds")
    assert out["transfer"].shape == (4, 4)
    # symmetry and diagonal = 1
    M = out["transfer"].values
    np.testing.assert_array_almost_equal(M, M.T)
    np.testing.assert_array_almost_equal(np.diag(M), np.ones(4))
    assert out["embedding"].shape == (4, 2)
    assert np.isfinite(out["embedding"].values).all()
    assert out["entropy"].shape == (4,)
    assert (out["entropy"] >= 0).all()


def test_compute_embeddings_spectral(D):
    g = EntropyTransferGraph(D)
    out = g.compute_embeddings(method="spectral")
    assert out["embedding"].shape == (4, 2)
    assert np.isfinite(out["embedding"].values).all()


def test_lambda_auto_positive(D):
    g = EntropyTransferGraph(D)
    g.compute_embeddings()
    assert g.lam > 0
