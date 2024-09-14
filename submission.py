import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl
import statsmodels.regression.linear_model as lm

# UTIL
# CONSTANTS
LIMIT = 10000
INF = 1000000000
COMM_RATE = 0.0010

currentPos = np.zeros(50)

# LEAD-LAG STRATEGY

memo = {}


def predict(currentPos, df, pct, ticker, indices, lags, deg):
    max_lag = np.max(lags)
    if ticker not in memo:
        response = pct[ticker].values[max_lag:]
        predictor = []
        for i in range(max_lag, pct.shape[0]):
            predictor.append(pct[indices].iloc[i - lags, :].values.T.flatten())

        result = lm.OLS(
            response,
            np.array(predictor)
        ).fit()

        memo[ticker] = result.params

    param = memo[ticker]

    predict_mu = np.sum(pct[indices].iloc[lags * -1, :].values.T.flatten() * param)
    val = df[ticker].values[-1]
    est = predict_mu
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    if (est * vol * val) > trans:
        currentPos[ticker] = vol
    elif (est * vol * val) < -trans:
        currentPos[ticker] = -vol
    else:
        pass


def getMyPosition(prices):
    global currentPos

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    pcf = df.pct_change().dropna()
    for i in range(50):
        predict(currentPos, df, pcf, i, list(range(50)), np.array([1, 286, 287]), 1)

    return currentPos
