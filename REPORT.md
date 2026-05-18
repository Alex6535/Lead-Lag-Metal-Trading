# Rapport — Lead-Lag Networks in Metal Trading

**Période** : 2020-01-01 → 2025-01-01 · **Fréquence** : daily · **8 actifs** · **1 305 observations**

---

## 1. Données & qualité

| Actif | Ticker | NaN bruts |
|---|---|---|
| Gold | GC=F | 47 |
| Silver | SI=F | 47 |
| Oil | CL=F | 47 |
| EURUSD | EURUSD=X | 0 |
| JPYUSD | JPYUSD=X | 0 |
| DXY | DX-Y.NYB | 47 |
| UST10Y | ^TNX | 47 |
| Bund10Y | IS0L.DE | 30 |

**Diagnostic missingness** : test ANOVA-by-pattern → **MCAR plausible** (p > 0.05 sur toutes les colonnes). Les trous correspondent aux jours fériés US différents des FX → asynchronicité de calendrier, pas de biais. Imputation **forward-fill** justifiée (les prix sont des stocks). Dataset final : `(1305, 8)`, 0 NaN.

Voir [outputs/01_missingness.png](outputs/01_missingness.png).

---

## 2. Preprocessing

- Log-returns : `(1302, 8)`, RobustScaler appliqué.
- **5 filtres comparés sur Gold** ([outputs/02_filters.png](outputs/02_filters.png)) :
  - Kalman (smooth optimal random-walk) — suit fidèlement le signal
  - Butterworth (low-pass) — supprime bien le HF
  - Savitzky-Golay — préserve les pics
  - MA(20) — référence
  - EMA TA-Lib — causal, utilisable en live
- **Markov-Switching** sur Gold : 2 régimes détectés → **1032 jours calmes / 272 jours stressés** (~21 % du temps). Cohérent avec les épisodes 2020 (COVID), 2022 (guerre Ukraine, hausse de taux Fed).

---

## 3. Corrélations linéaires (log-returns)

|  | Gold | Silver | Oil | EURUSD | JPYUSD | DXY | UST10Y | Bund10Y |
|---|---|---|---|---|---|---|---|---|
| **Gold** | 1.00 | **0.78** | 0.14 | 0.06 | 0.01 | **−0.38** | −0.23 | 0.24 |
| **Silver** | 0.78 | 1.00 | 0.19 | 0.05 | 0.00 | −0.38 | −0.09 | 0.14 |
| **EURUSD** | 0.06 | 0.05 | −0.04 | 1.00 | **0.45** | −0.09 | −0.07 | 0.08 |
| **UST10Y** | −0.23 | −0.09 | 0.15 | −0.07 | −0.06 | 0.23 | 1.00 | **−0.46** |

**Lectures** :
- Or/Argent fortement liés (0.78) — cluster métaux précieux confirmé.
- Métaux ↔ DXY anticorrélés (−0.38) — relation classique : USD fort = or faible.
- UST10Y ↔ Bund10Y inversés (−0.46) — divergence Fed/BCE sur la période.
- FX EURUSD/JPYUSD modérément liés (0.45) — facteur USD commun.

---

## 4. Lead-Lag DTW (cœur du projet)

### Matrice de distances DTW (proximité de forme)
Plus la distance est petite, plus les dynamiques sont alignées :

| Paire | Distance |
|---|---|
| **JPYUSD ↔ Bund10Y** | **9.25** ⭐ |
| DXY ↔ UST10Y | 18.22 |
| Gold ↔ Silver | 18.56 |
| EURUSD ↔ JPYUSD | 22.31 |
| EURUSD ↔ Bund10Y | 24.14 |

→ **Trois clusters naturels émergent** :
1. **Métaux** : Gold–Silver
2. **USD-driven** : DXY–UST10Y
3. **EUR/JPY/Bund** : flux JPY-Bund spectaculairement proches (carry trade JPY → Bund)

### Matrice lead-lag (ligne mène colonne si > 0)

Lectures saillantes :
| Leader | Follower | Lag (jours) |
|---|---|---|
| **Gold** | Bund10Y | +7.0 |
| **Gold** | JPYUSD | +6.3 |
| **Gold** | DXY | +4.7 |
| **JPYUSD** | Oil | +12.4 |
| **DXY** | EURUSD | +11.9 |
| **UST10Y** | Bund10Y | +11.2 |

**Insights** :
- **Gold est un leader généralisé** : tous les lags sur sa ligne sont positifs sauf vs Silver. L'or anticipe les mouvements de Bund, JPY et DXY de ~5-7 jours.
- **UST10Y mène le Bund de ~11 jours** : la Fed donne le tempo, la BCE suit (cohérent avec le cycle 2022-2024).
- **DXY mène EURUSD de ~12 jours** : effet mécanique (DXY pondéré à 57 % par EUR) + persistance des flux USD.

CSV complets : [outputs/04_dtw_distance.csv](outputs/04_dtw_distance.csv) · [outputs/04_lag_matrix.csv](outputs/04_lag_matrix.csv)

---

## 5. Tuning & Validation

- **Optuna** (10 essais, expanding-window CV 5 folds) sur la Sakoe-Chiba band pour la paire `Gold → Silver`.
- **Best radius** : `26` · **Best RMSE CV** : `1955.4`
- Validation full-sample : `lag=2 jours`, MSE=3.74e6, RMSE=1933, Wasserstein=1915.

⚠️ Le forecast naïf "shift" est ici un **plancher** (baseline). Les RMSE absolus sont élevés car on prédit le **prix de Silver** avec celui de Gold décalé — attendu sans recalibration de niveau. La valeur du test est **comparative** (le tuning montre que l'optimum est non trivial : radius=26 ≠ valeur par défaut), pas absolue.

Une amélioration directe : prédire les **log-returns standardisés** plutôt que les prix bruts.

---

## 6. Entropy Transfer Graph

**Pipeline mathématique** : D → S = exp(−λD), λ auto = 0.0134 → P (row-norm) → H Shannon → M = 1−JSD → MDS 2D → graphe.

### Entropies de Shannon (faible = leader marqué)

| Rang | Actif | H |
|---|---|---|
| 1 ⭐ | **JPYUSD** | 2.008 |
| 2 | Bund10Y | 2.011 |
| 3 | DXY | 2.024 |
| 4 | UST10Y | 2.026 |
| 5 | EURUSD | 2.030 |
| 6 | Gold | 2.039 |
| 7 | Oil | 2.039 |
| 8 | **Silver** | 2.046 |

**Lecture** : les entropies sont resserrées (2.01-2.05) → tous les actifs partagent une part d'information commune. Néanmoins :
- **JPYUSD et Bund10Y sont les "transmetteurs" les plus marqués** : leurs profils de similarité sont les plus concentrés → ils servent d'**indicateurs avancés** du système.
- **Silver est le plus "réactif"** : son profil est le plus diffus → consommateur d'information.

Voir [outputs/06_entropy_graph.png](outputs/06_entropy_graph.png) · [outputs/06_transfer_matrix.csv](outputs/06_transfer_matrix.csv)

---

## 7. Conclusion & implications

### Pour le trading
1. **Gold comme leading indicator** : ses signaux précèdent Bund/JPY/DXY de 5-7 jours → utilisable comme filtre directionnel sur les FX safe-haven.
2. **Cluster JPY-Bund** : l'interconnexion la plus forte du dataset (carry trade structurel) — toute rupture de ce lien serait un signal macro fort.
3. **UST10Y → Bund10Y (~11 jours)** : la transmission Fed→BCE est exploitable sur les spreads de taux.

### Pour le risk management
- **Nœuds centraux à surveiller** : JPYUSD et Bund10Y (faible entropie = forte capacité de propagation de chocs).
- **Régimes Markov** : 21 % du temps en état stressé → calibrer le risk budget en conséquence.
- **Diversification réelle** : Oil est l'actif le plus indépendant (corrélations < 0.2 partout) → vrai bénéfice de diversification.

### Limites & pistes
- Les RMSE absolus (Tâche 5) ne valident que le pipeline, pas une stratégie tradable. Refactor : prédire les **returns**, ajouter un **modèle de calibration** (régression linéaire sur le shifted leader).
- DTW est **symétrique** par construction : la directionalité vient seulement du warping path. Pour une vraie causalité, implémenter la **Transfer Entropy de Schreiber**.
- Les entropies très resserrées suggèrent d'**augmenter λ** (plus de discrimination) ou de passer en **DTW multi-dimensionnel** (MDPI 2022) pour capter les co-mouvements.
- Tester sur fréquence **horaire** (yfinance limite à 730 jours) pour capter les lead-lags intra-day plus exploitables en trading systématique.

---

**Reproductibilité** : `python notebook.py` · **Tests** : 20/20 verts · **Code** : [src/](src/)
