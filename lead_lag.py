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
def predict(currentPos, df, ticker, indices, lags, deg=1):

    result = ardl.ARDL(
        df[ticker].pct_change().dropna().values[-600:],
        0,
        df[indices].pct_change().dropna().values[-600:],
        lags,
        causal=True
    ).fit()

    predict_mu = result.forecast()
    est = predict_mu
    val = df[ticker].values[-1]
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

    # for i in range(42):
    #     if i in [19,27,38]:
    #         continue

    #     predict(currentPos, df, i, [i+8], np.arange(572,575), 1)

    # for i in range(42,50):
    #     predict(currentPos, df, i, [i-42], np.arange(571,574), 1)

    # predict(currentPos, df, 19, [48], np.arange(286, 289), 1)

    predict(currentPos, df, 7, [36], [287])
 

    return currentPos