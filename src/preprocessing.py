import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pykalman import KalmanFilter
from scipy.signal import butter, filtfilt, savgol_filter
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from prophet import Prophet

class Preprocessing:
    def __init__(self, data: pd.DataFrame):
        # Les prix bruts nettoyés qu'on reçoit du resampled/merged
        self.data = data.copy()
        
        self.scaled = None
        self.log_returns = None

    def transform_data(self, ma_window: int = 20) -> dict:
        """
        Génère les log-returns, moy. mobile et une version scalée au besoin.
        Le log return stationnarise bien la variance (utile pour le DTW/Corrélation ensuite).
        """
        prices = self.data.astype(float)
        
        # Le premier jour de log_ret sera NaN, c'est normal
        log_ret = np.log(prices / prices.shift(1)).dropna()
        ma = prices.rolling(ma_window, min_periods=1).mean()
        
        # RobustScaler pour pas se faire exploser par les outliers (crise covid etc)
        scaler = RobustScaler()
        scaled_vals = scaler.fit_transform(prices.values)
        scaled = pd.DataFrame(scaled_vals, index=prices.index, columns=prices.columns)
        
        self.log_returns = log_ret
        self.scaled = scaled
        
        return {"log_returns": log_ret, "moving_average": ma, "scaled": scaled}

    def apply_kalman_filter(self, col: str) -> pd.Series:
        """
        Filtre de Kalman simple (random walk local).
        Pas de paramètre de fenêtre hyper chiant à optimiser.
        """
        s = self.data[col].dropna().astype(float)
        
        # Init de base, matrices à 1 pour un random walk
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=s.iloc[0],
            initial_state_covariance=1.0,
            observation_covariance=1.0,
            transition_covariance=0.01,
        )
        state_means, _ = kf.smooth(s.values)
        return pd.Series(state_means.ravel(), index=s.index, name=f"{col}_kalman")

    def apply_butterworth_filter(self, col: str, cutoff: float = 0.1, order: int = 3) -> pd.Series:
        # Filtre low-pass standard. Cutoff 0.1 évite le bruit HF journalier.
        s = self.data[col].dropna().astype(float)
        b, a = butter(order, cutoff, btype="low")
        y = filtfilt(b, a, s.values)
        return pd.Series(y, index=s.index, name=f"{col}_butter")

    def apply_savgol_filter(self, col: str, window: int = 21, poly: int = 3) -> pd.Series:
        # Savitzky-Golay, pratique pour lisser sans aplatir complétement les pics de prix
        s = self.data[col].dropna().astype(float)
        if window % 2 == 0:
            window += 1 # Savgol a besoin d'une impaire
        y = savgol_filter(s.values, window_length=window, polyorder=poly)
        return pd.Series(y, index=s.index, name=f"{col}_savgol")

    def apply_moving_average(self, col: str, window: int = 20) -> pd.Series:
        # La bonne vieille moyenne mobile (le baseline)
        return self.data[col].rolling(window, min_periods=1).mean().rename(f"{col}_ma")

    def apply_ta_lib_filter(self, col: str, window: int = 20) -> pd.Series:
        """EMA via talib si installé, sinon pandas fall-back"""
        s = self.data[col].dropna().astype(float)
        try:
            import talib
            y = talib.EMA(s.values, timeperiod=window)
            return pd.Series(y, index=s.index, name=f"{col}_ema")
        except ImportError:
            # Fallback direct
            return s.ewm(span=window, adjust=False).mean().rename(f"{col}_ema")

    def apply_all_filters(self, col: str) -> pd.DataFrame:
        """Helper pour plot tous les filtres d'un coup."""
        return pd.concat([
            self.data[col].rename(f"{col}_raw"),
            self.apply_kalman_filter(col),
            self.apply_butterworth_filter(col),
            self.apply_savgol_filter(col),
            self.apply_moving_average(col),
            self.apply_ta_lib_filter(col),
        ], axis=1)

    def detect_regimes_markov(self, col: str, k_regimes: int = 2) -> pd.Series:
        """
        Détection des régimes via Markov (par ex: normal vs volatil).
        """
        # On fit sur les log returns c'est beaucoup plus stable que les prix 
        ret = np.log(self.data[col]).diff().dropna()
        
        # switching_variance=True car c'est la volat qui caractérise le plus un stress market
        model = MarkovRegression(ret, k_regimes=k_regimes, trend="c", switching_variance=True)
        res = model.fit(disp=False)
        regimes = res.smoothed_marginal_probabilities.idxmax(axis=1)
        return regimes.rename(f"{col}_regime")

    def detect_outliers_prophet(self, col: str, interval_width: float = 0.99) -> pd.DataFrame:
        # Check les outliers avec prophet (rapide et pas mal sur du daily)
        s = self.data[col].dropna()
        df = pd.DataFrame({"ds": s.index, "y": s.values})
        m = Prophet(interval_width=interval_width, daily_seasonality=False)
        m.fit(df)
        fc = m.predict(df[["ds"]])
        
        merged = df.merge(fc[["ds", "yhat_lower", "yhat_upper"]], on="ds")
        # Est-ce que le point réel sort de l'intervalle de confiance ?
        merged["outlier"] = (merged["y"] < merged["yhat_lower"]) | (merged["y"] > merged["yhat_upper"])
        return merged.set_index("ds")
