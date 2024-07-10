import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl

# UTIL
# CONSTANTS
LIMIT = 10000
INF = 1000000000
COMM_RATE = 0.0010

currentPos = np.zeros(50)


# LEAD-LAG STRATEGY
def predict(currentPos, df, ticker, indices, lags, deg, shift=0):
    if df[ticker].shape[0] < 201 + shift:
        return

    matchedSeries = df[indices].pct_change().dropna().values
    # print(matchedSeries.shape)
    if len(indices) > 1:
        end = matchedSeries.shape[0]
        matchedSeries = matchedSeries[(end - 200 - shift):(end - shift), :]
    else:
        end = len(matchedSeries)
        matchedSeries = matchedSeries[(end - 200 - shift):(end - shift)]
    # print(matchedSeries)

    result = ardl.ARDL(
        df[ticker].pct_change().dropna().values[-200:],
        0,
        matchedSeries,
        lags,
        causal=True
    ).fit()

    predict_mu = result.forecast()
    est = predict_mu
    val = df[ticker].values[-1]
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    pforecast = est * val
    # print("Predict", pforecast)
    # print("Previous actual", df[ticker].values[-1] - df[ticker].values[-2])
    if abs(vol * pforecast) > trans:
        if pforecast > 0:
            currentPos[ticker] = vol
        else:
            currentPos[ticker] = -vol


def getMyPosition(prices):
    global currentPos, oldPos

    nins, nt = prices.shape

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    predict(currentPos, df, 38, list(range(50)), 1, 1.1)
    return currentPos


