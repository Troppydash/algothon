import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl

# UTIL
# CONSTANTS
LIMIT = 10000
INF = 1000000000
COMM_RATE = 0.0010

currentPos = np.zeros(50)

memo = {}


# LEAD-LAG STRATEGY
def predict(currentPos, df, pct, ticker, indices, lags, deg):
    if ticker not in memo:
        result = ardl.ARDL(
            df[ticker].pct_change().dropna().values[:],
            0,
            df[indices].pct_change().dropna().values[:],
            lags,
            causal=True
        ).fit()

        memo[ticker] = result.params

    param = memo[ticker]

    predict_mu = np.sum(pct[indices].iloc[lags * -1, :].values.T.flatten() * param[1:]) + param[0]
    val = df[ticker].values[-1]
    est = predict_mu
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    pforecast = est * val
    if abs(vol * pforecast) > trans:
        if pforecast > 0:
            currentPos[ticker] = vol
        else:
            currentPos[ticker] = -vol


def getMyPosition(prices):
    global currentPos

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    pcf = df.pct_change().dropna()
    for i in range(50):
        if i in [2, 6, 16, 27, 38, 39, 48]:
            continue

        predict(currentPos, df, pcf, i, list(range(50)), np.array([1, 286, 287, 288]), 1)

    return currentPos