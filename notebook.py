"""Lead-Lag Networks in Metal Trading — end-to-end runnable pipeline.

Usage
-----
    python notebook.py

Each "# %% Task N — ..." block mirrors a notebook cell. Outputs (figures,
tables) are saved to ``outputs/`` so the script stays headless-friendly.
"""
from __future__ import annotations

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

from src.config import TICKERS
from src.data_loader import DataLoader
from src.preprocessing import Preprocessing
from src.eda import EDA
from src.lead_lag_dtw import LeadLagDTW
from src.tuning import ModelTuningValidation
from src.entropy_graph import EntropyTransferGraph


def banner(msg: str) -> None:
    print("\n" + "=" * 70 + f"\n  {msg}\n" + "=" * 70)


# %% Task 1 — DataLoader -----------------------------------------------------
banner("Task 1 — DataLoader")
dl = DataLoader(TICKERS, start_date="2020-01-01", end_date="2025-01-01", freq="1d")
dl.fetch_data()
merged = dl.merge_data()
print(f"merged shape: {merged.shape}")
print(f"missing per col:\n{merged.isna().sum()}")

fig = dl.eda_missing(merged)
fig.savefig(os.path.join(OUT, "01_missingness.png"), dpi=120, bbox_inches="tight")
plt.close(fig)

mcar = dl.missing_test(merged)
print(f"MCAR plausible: {mcar['mcar_plausible']}")

prices = dl.impute_data(merged)
prices.to_csv(os.path.join(OUT, "01_prices.csv"))
print(f"prices clean: {prices.shape}, NaN={prices.isna().sum().sum()}")


# %% Task 2 — Preprocessing --------------------------------------------------
banner("Task 2 — Preprocessing")
pp = Preprocessing(prices)
feats = pp.transform_data(ma_window=20)
print(f"log_returns: {feats['log_returns'].shape}")

target = "Gold" if "Gold" in prices.columns else prices.columns[0]
filt = pp.apply_all_filters(target)
ax = filt.plot(figsize=(12, 5), title=f"{target} — comparaison des 5 filtres")
ax.figure.savefig(os.path.join(OUT, "02_filters.png"), dpi=120, bbox_inches="tight")
plt.close(ax.figure)

try:
    regimes = pp.detect_regimes_markov(target)
    print(f"Markov regimes counts:\n{regimes.value_counts()}")
except Exception as e:
    print(f"[skip] Markov regimes: {e}")


# %% Task 3 — EDA ------------------------------------------------------------
banner("Task 3 — EDA")
eda = EDA(prices)
corr = eda.correlation_matrix(max_lag=5)
print("Correlation matrix:")
print(corr["corr"].round(2))
corr["corr"].to_csv(os.path.join(OUT, "03_corr.csv"))

g, D_eda = eda.dtw_clustermap()
g.fig.savefig(os.path.join(OUT, "03_dtw_clustermap.png"), dpi=120, bbox_inches="tight")
plt.close(g.fig)
D_eda.to_csv(os.path.join(OUT, "03_dtw_distance.csv"))


# %% Task 4 — LeadLagDTW -----------------------------------------------------
banner("Task 4 — LeadLagDTW")
ll = LeadLagDTW(prices, sakoe_chiba_radius=20)
res = ll.identify_lead_lag()
print("DTW distance matrix:")
print(res["distance"].round(2))
print("\nLead-lag matrix (row leads col if > 0):")
print(res["lag"].round(2))
res["distance"].to_csv(os.path.join(OUT, "04_dtw_distance.csv"))
res["lag"].to_csv(os.path.join(OUT, "04_lag_matrix.csv"))


# %% Task 5 — Tuning & Validation -------------------------------------------
banner("Task 5 — Tuning & Validation")
cols = list(prices.columns)
leader, follower = cols[0], cols[1]
mt = ModelTuningValidation(prices, leader=leader, follower=follower, n_splits=5)
tune = mt.tune_model(n_trials=10)
print(f"best params: {tune['best_params']}  best RMSE: {tune['best_value']:.4f}")
print(f"validation: {mt.validate_model()}")


# %% Task 6 — EntropyTransferGraph -------------------------------------------
banner("Task 6 — EntropyTransferGraph")
etg = EntropyTransferGraph(res["distance"])
emb = etg.compute_embeddings(method="mds")
print(f"lambda auto: {emb['lambda']:.4f}")
print("Shannon entropies (low = strong leader):")
print(emb["entropy"].sort_values().round(3))
emb["transfer"].to_csv(os.path.join(OUT, "06_transfer_matrix.csv"))
emb["entropy"].to_csv(os.path.join(OUT, "06_entropy.csv"))

fig, ax = plt.subplots(figsize=(9, 7))
etg.plot_graph(threshold=float(np.median(emb["transfer"].values)), ax=ax)
fig.savefig(os.path.join(OUT, "06_entropy_graph.png"), dpi=120, bbox_inches="tight")
plt.close(fig)


# %% Task 7 — Conclusion -----------------------------------------------------
banner("Task 7 — Conclusion")
ent = emb["entropy"].sort_values()
print(f"Strongest leader (lowest entropy): {ent.index[0]}")
print(f"Most reactive (highest entropy):   {ent.index[-1]}")
print(f"\nAll outputs written to: {OUT}")
