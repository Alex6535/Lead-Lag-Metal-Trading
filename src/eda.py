"""EDA — interactive plots, correlation, DTW clustermap, seasonality.

References
----------
- Bokeh user guide: https://docs.bokeh.org
- dtaidistance DTW: https://dtaidistance.readthedocs.io
- statsmodels STL: https://www.statsmodels.org/dev/generated/statsmodels.tsa.seasonal.STL.html
"""
import numpy as np
import pandas as pd

class EDA:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def plot_timeseries(self):
        """Plot propre avec bokeh pour zoomer/scroll en synchro"""
        from bokeh.layouts import column
        from bokeh.plotting import figure

        figs = []
        x_range = None
        for col in self.data.columns:
            p = figure(
                x_axis_type="datetime",
                title=col,
                height=180,
                width=900,
                x_range=x_range,
            )
            p.line(self.data.index, self.data[col].values, line_width=1.2)
            # Lie le x_range pour que tous les graphes zooment en même temps
            x_range = x_range or p.x_range
            figs.append(p)
        return column(*figs)

    def correlation_matrix(self, max_lag: int = 5) -> dict:
        """
        Cross-corr classique. Pas fifou parce que le lag est constant,
        mais ça donne une idée de base avant le DTW.
        """
        ret = np.log(self.data).diff().dropna()
        corr = ret.corr()
        cross = {}
        
        cols = ret.columns
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                xs = []
                for lag in range(1, max_lag + 1):
                    # a.corr(b shifté)
                    xs.append(ret[a].corr(ret[b].shift(lag)))
                cross[(a, b)] = xs
        return {"corr": corr, "cross_corr": cross}

    def dtw_distance_matrix(self) -> pd.DataFrame:
        from dtaidistance import dtw

        ret = np.log(self.data).diff().dropna()
        # Normalisation Z-score vitale ici, sinon le DTW compare des niveaux et pas les "formes"
        z = (ret - ret.mean()) / ret.std()
        
        cols = list(z.columns)
        n = len(cols)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = dtw.distance(z[cols[i]].values, z[cols[j]].values)
                D[i, j] = D[j, i] = d
        return pd.DataFrame(D, index=cols, columns=cols)

    def dtw_clustermap(self):
        import seaborn as sns

        D = self.dtw_distance_matrix()
        g = sns.clustermap(D, cmap="viridis", annot=True, fmt=".1f")
        return g, D

    def seasonality_tracker(self, period: int = 5) -> dict:
        """Decomp STL pour trouver la saisonnalité intra-hebbdomadaire (5 jours d'open)"""
        from statsmodels.tsa.seasonal import STL

        out = {}
        for col in self.data.columns:
            s = self.data[col].dropna()
            if len(s) < 2 * period + 1:
                continue
            stl = STL(s, period=period, robust=True).fit()
            out[col] = {"trend": stl.trend, "seasonal": stl.seasonal, "resid": stl.resid}
        return out
