"""ModelTuningValidation — Optuna search + expanding-window CV.

Metrics
-------
- MSE / MAE / RMSE: classical pointwise errors.
- Wasserstein distance (1D, scipy.stats.wasserstein_distance): optimal-transport
  cost between forecast and actual *distributions* — sensitive to tail shape,
  not just mean error. Useful when residuals are heavy-tailed (financial data).

CV
--
- ``sklearn.model_selection.TimeSeriesSplit`` → expanding window, no shuffle.
"""
import numpy as np
import pandas as pd
from src.lead_lag_dtw import LeadLagDTW

class ModelTuningValidation:
    def __init__(self, data: pd.DataFrame, leader: str, follower: str, n_splits: int = 5):
        """
        Hyperparam tuning via Optuna et validation par TimeSeriesSplit (expanding window).
        Surtout pas de KFold classique sinon on lit le futur pour prédire le passé.
        """
        self.data = data.copy()
        self.leader = leader
        self.follower = follower
        self.n_splits = n_splits
        self.best_params_ = {}

    @staticmethod
    def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
        """
        Metrics classiques + Wasserstein (très cool pour comparer les lois de distrib
        des erreurs, parce qu'en finance les queues de distrib sont fat).
        """
        from scipy.stats import wasserstein_distance

        df = pd.concat([y_true, y_pred], axis=1).dropna()
        df.columns = ["y", "yhat"]
        if df.empty:
            return {"mse": np.nan, "mae": np.nan, "rmse": np.nan, "wass": np.nan}
            
        err = df["y"] - df["yhat"]
        mse = float((err ** 2).mean())
        mae = float(err.abs().mean())
        
        return {
            "mse": mse,
            "mae": mae,
            "rmse": float(np.sqrt(mse)),
            "wass": float(wasserstein_distance(df["y"].values, df["yhat"].values)),
        }

    def _cv_score(self, radius: int) -> float:
        """L'expanding window RMSE score"""
        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        scores = []
        idx = np.arange(len(self.data))
        
        for train_idx, test_idx in tscv.split(idx):
            train = self.data.iloc[train_idx]
            test = self.data.iloc[test_idx]
            
            # On fit le DTW sur le passsé
            ll = LeadLagDTW(train[[self.leader, self.follower]], sakoe_chiba_radius=radius)
            ll.identify_lead_lag()
            lag = int(round(ll.lag_matrix_.loc[self.leader, self.follower]))
            lag = max(lag, 0)
            
            # On prédit le "futur" (test) statiquement
            yhat = test[self.leader].shift(lag)
            scores.append(self._metrics(test[self.follower], yhat)["rmse"])
            
        return float(np.nanmean(scores))

    def tune_model(self, n_trials: int = 20, radius_range=(2, 30)) -> dict:
        """Optuna TPE classique pour trouver la bande de Sakoe-Chiba parfaite"""
        import optuna

        # On mute Optuna sinon ça spam la console à chaque boucle
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            r = trial.suggest_int("sakoe_chiba_radius", radius_range[0], radius_range[1])
            return self._cv_score(r)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params_ = study.best_params
        
        return {"best_params": study.best_params, "best_value": study.best_value, "study": study}

    def validate_model(self, radius: int = None) -> dict:
        """Le run final sur tout le dataset avec le best hyperparam"""
        r = radius if radius is not None else self.best_params_.get("sakoe_chiba_radius", 10)
        
        ll = LeadLagDTW(self.data[[self.leader, self.follower]], sakoe_chiba_radius=r)
        ll.identify_lead_lag()
        lag = int(round(ll.lag_matrix_.loc[self.leader, self.follower]))
        lag = max(lag, 0)
        
        yhat = self.data[self.leader].shift(lag)
        return {"lag": lag, "radius": r, **self._metrics(self.data[self.follower], yhat)}
