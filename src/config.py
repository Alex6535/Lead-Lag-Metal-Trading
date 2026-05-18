"""
Setup des tickers.
Pourquoi IS0L.DE pour le Bund ? 
Parce que les yields purs sur Yahoo c'est l'enfer à fetch proprement, 
cet ETF suit l'index Bloomberg Germany, ça fait parfaitement l'affaire comme proxy du fixed income EUR.
"""

TICKERS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Oil": "CL=F",
    "EURUSD": "EURUSD=X",
    "JPYUSD": "JPYUSD=X",
    "DXY": "DX-Y.NYB",
    "UST10Y": "^TNX",
    "Bund10Y": "IS0L.DE",
}
