"""DataLoader — fetch, merge, diagnose and impute Yahoo Finance time series.

References
----------
- yfinance docs: https://github.com/ranaroussi/yfinance
- Little's MCAR test (Little, 1988): EM-based likelihood-ratio chi-square test.
  See https://github.com/RianneSchouten/pyampute and
  https://medium.com/@tarangds/understanding-littles-mcar-test-a-key-tool-in-missing-data-analysis-47fd70698149
- Forward-fill rationale for prices: prices are *stocks* (last observed value
  remains valid until a new trade), unlike returns which are flows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats

class DataLoader:
    def __init__(self, tickers: dict, start_date: str, end_date: str, freq: str = "1d"):
        """
        Gère le téléchargement Yahoo Finance, la fusion des actifs et l'imputation.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.freq = freq
        
        self.raw_data = {}
        self.df = None

    def fetch_data(self) -> dict:
        """
        Télécharge chaque ticker. Tolère si une API call plante.
        """
        out = {}
        for name, sym in self.tickers.items():
            try:
                # auto_adjust=False pour garder le close classique
                d = yf.download(
                    sym,
                    start=self.start_date,
                    end=self.end_date,
                    interval=self.freq,
                    progress=False,
                    auto_adjust=False,
                )
                
                if d is None or d.empty:
                    print(f"Skipping {name} (empty)")
                    continue
                
                # Gestion du multi-index yfinance récent
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                    
                series = d["Close"].rename(name)
                
                # J'enlève la timezone pour éviter les pbs pd.concat plus tard
                if series.index.tz is not None:
                    series.index = series.index.tz_convert("UTC").tz_localize(None)
                    
                out[name] = series
            except Exception as e:
                print(f"Erreur sur {name} : {e}")
                
        self.raw_data = out
        return out

    def merge_data(self) -> pd.DataFrame:
        """Join outer sur la date"""
        if not self.raw_data:
            raise ValueError("Faut appeler fetch_data() avant")
            
        merged = pd.concat(self.raw_data.values(), axis=1, join="outer").sort_index()
        merged.columns = list(self.raw_data.keys())
        self.df = merged
        return merged

    def eda_missing(self, df: pd.DataFrame = None):
        """Plot missingno classique"""
        df = df if df is not None else self.df
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        msno.matrix(df, ax=axes[0], sparkline=False)
        axes[0].set_title("Matrice des NAs")
        msno.heatmap(df, ax=axes[1])
        axes[1].set_title("Corrélation des NAs")
        plt.tight_layout()
        
        return fig

    def missing_test(self, df: pd.DataFrame = None) -> dict:
        """
        Test proxy de Little (ANOVA sur les groupes de missing patterns).
        Si p-value < 0.05, rejette l'hypothèse que c'est manquant complètement au hasard.
        """
        df = df if df is not None else self.df
        patterns = df.isna().apply(lambda r: tuple(r.values), axis=1)
        
        res = {}
        for col in df.columns:
            obs = df[col].dropna()
            # on regarde les patterns des autres colonnes quand col est observée
            grp_labels = patterns.loc[obs.index]
            groups = [obs.loc[grp_labels == p].values for p in grp_labels.unique()]
            
            # filtre les singletons
            groups = [g for g in groups if len(g) > 1]
            
            if len(groups) < 2:
                res[col] = {"F": np.nan, "p": np.nan}
                continue
                
            F, p = stats.f_oneway(*groups)
            res[col] = {"F": F, "p": p}
            
        ps = [r["p"] for r in res.values() if not np.isnan(r["p"])]
        
        # MCAR si on a pas de preuves du contraire (p > 5%)
        return {
            "per_column": res,
            "mcar_plausible": len(ps) > 0 and all(p > 0.05 for p in ps)
        }

    def impute_data(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Forward fill c'est ok pour les prix (le prix reste valide).
        Backfill juste pour les extrêmes débuts.
        """
        df = df if df is not None else self.df
        clean = df.ffill().bfill()
        self.df = clean
        return clean
