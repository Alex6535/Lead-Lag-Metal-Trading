# ROADMAP — Lead-Lag Metal Trading

Workflow obligatoire pour CHAQUE tâche :
1. **Recherche** (WebSearch) → comprendre la technique, noter les pièges
2. **Design POO** → définir classe, attributs, méthodes, types
3. **Implémentation** dans `src/`
4. **Test** unitaire + sanity check visuel
5. **Markdown narratif** dans le notebook (formules + justification)
6. **Commit** atomique

---

## Phase 0 — Setup
- [ ] `src/`, `tests/`, `notebook.ipynb`, `requirements.txt`
- [ ] Vérifier proxy Bund 10y via WebSearch
- [ ] Configurer `pytest` + fixture de données mockées

## Phase 1 — DataLoader (`src/data_loader.py`)
**Recherche** : yfinance hourly limits (730j max), gestion fuseaux, MCAR test (Little 1988).
- [ ] `DataLoader(tickers, start, end, freq)`
- [ ] `fetch_data()` — try/except par ticker, log les fails
- [ ] `merge_data()` — outer join sur index datetime UTC
- [ ] `eda_missing()` — `msno.matrix` + `msno.heatmap`
- [ ] `missing_test()` — Little's MCAR (statsmodels ou impl manuelle)
- [ ] `impute_data()` — ffill (justifier : prix = stock, pas flux)
- **Tests** : shapes, pas de NaN après impute, dates monotones, tous tickers présents.

## Phase 2 — Preprocessing (`src/preprocessing.py`)
**Recherche** : Kalman pour prix (random walk + noise), Butterworth cutoff pour daily, Savgol window/polyorder, RobustScaler vs Standard pour returns.
- [ ] `Preprocessing(data)`
- [ ] `transform_data()` — log-returns, MA, RobustScaler
- [ ] `apply_kalman_filter()` (pykalman, local level model)
- [ ] `apply_butterworth_filter()` (scipy, lowpass)
- [ ] `apply_savgol_filter()`
- [ ] `apply_moving_average()`
- [ ] `apply_ta_lib_filter()` (ex EMA, KAMA)
- [ ] `anomaly_detection()` — `MarkovRegression` (statsmodels) + Prophet
- [ ] **Comparaison visuelle des 5 filtres sur Gold** (subplot 5x1) + tableau MSE vs raw.
- **Tests** : longueur préservée, pas de leak (filtres causaux où possible).

## Phase 3 — EDA (`src/eda.py`)
**Recherche** : DTW clustermap (tslearn), STL params (period daily=5 jours ouvrés ?).
- [ ] `plot_timeseries()` — Bokeh, panneaux liés
- [ ] `correlation_matrix()` — Pearson + cross-corr lags 1-5
- [ ] `dtw_clustermap()` — matrice DTW pairwise, `seaborn.clustermap`
- [ ] `seasonality_tracker()` — STL par actif
- **Tests** : matrice symétrique, diagonale = 0, clusters reproductibles (seed).

## Phase 4 — LeadLagDTW (`src/lead_lag_dtw.py`) ⭐ COEUR
**Recherche obligatoire** : warping path interpretation (au-dessus diag = série Y en retard), Sakoe-Chiba band, FastDTW vs DTW exact.
- [ ] `compute_dtw(s1, s2)` → (distance, path)
- [ ] `_path_lag(path)` → lag moyen signé (déviation médiane à la diagonale)
- [ ] `identify_lead_lag()` → matrice n×n de lags + matrice de distances
- [ ] `forecast(leader, follower, lag)` — naive shift forecast
- [ ] `validate(forecasts, actuals)` → MSE + directional accuracy
- **Tests** : sur séries synthétiques (sin + sin décalé de k) → vérifier lag retrouvé ≈ k.

## Phase 5 — ModelTuningValidation (`src/tuning.py`)
**Recherche** : Optuna TPE, expanding window CV pour TS, Wasserstein 1D.
- [ ] `tune_model()` — Optuna : window_size, sakoe_chiba_radius, filter params
- [ ] `validate_model()` — MSE/MAE/RMSE + `wasserstein_distance`
- [ ] CV expanding window (sklearn `TimeSeriesSplit`)
- [ ] Plots diagnostiques : study history, param importance, error dist
- **Tests** : pas de shuffle, train_end < test_start sur tous les folds.

## Phase 6 — EntropyTransferGraph (`src/entropy_graph.py`) ⭐ ORIGINALITÉ
**Recherche** : JSD formule (symétrique, bornée [0, log2]), MDS classique vs SMACOF, Spectral Embedding sur Laplacien normalisé.
- [ ] `compute_embeddings()` :
  1. S = exp(-λ D) — λ choisi tel que median(S) ≈ 0.5
  2. p_i = row-normalize(S)
  3. H(i) = -Σ p log p
  4. M_ij = 1 - JSD(p_i, p_j)
  5. Δ = 1 - M → MDS 2D et Spectral Embedding
- [ ] `plot_graph()` — networkx + bokeh, taille nœud = centralité (eigenvector)
- **Tests** : M symétrique, M_ii = 1, embeddings shape (n, 2), tous finis.

## Phase 7 — Conclusion (notebook markdown)
- [ ] Tableau récap : leaders / followers
- [ ] Clusters identifiés (métaux vs FX vs taux ?)
- [ ] Implications trading + risk
- [ ] Limites + pistes (transfer entropy de Schreiber, GNN, intraday tick data)

---

## Definition of Done (par phase)
✅ Code POO dans `src/` · ✅ Tests passent · ✅ Markdown narratif · ✅ Plots dans le notebook · ✅ Sources citées · ✅ Commit
