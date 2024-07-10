import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl

# UTIL
# CONSTANTS
LIMIT = 10000
INF = 1000000000
COMM_RATE = 0.0010

currentPos = np.zeros(50)
oldPos = np.zeros(50)

# Util functions
# Util functions to keep track of profit (cash + position value) for each ticker
# Need to be called once every iteration to ensure updated
tickers__cash = np.zeros(50)
tickers__posVal = np.zeros(50)


def trackTickerProfit(prices):
    global currentPos, oldPos
    changePost = currentPos - oldPos
    for i in range(50):
        comm = COMM_RATE * abs(changePost[i]) * prices[i][-1]
        tickers__cash[i] -= (comm + changePost[i] * prices[i][-1])
        tickers__posVal[i] = currentPos[i] * prices[i][-1]


def clampLimit(currentPos, df, ticker):
    # check for limits
    i = ticker
    pos, val = currentPos[i], df[i].values[-1]
    if pos * val > 10000:
        currentPos[i] = int(10000 / val)
    if pos * val < -10000:
        currentPos[i] = int(-10000 / val)

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

    if len(df[0]) == 700:
        startIndex = 500
        endIndex = 699

        outputStr = ""
        outputStr += str(df[0][699]) + "|"
        
        start = df[:].values[startIndex]
        end = df[:].values[endIndex]
        diff = end - start

        for i in range(50):
            if diff[i] < 0:
                diff[i] += start[i] * 2 * COMM_RATE
            else:
                diff[i] -= start[i] * 2 * COMM_RATE

        profit = ""
        for i in range(50):
            if abs(diff[i]) > 4:
                profit += f"{i}:{int(diff[i] * 10)/10:.1f};"
        outputStr += profit
        raise Exception(outputStr)

def getMyPosition(prices):
    global currentPos, oldPos

    nins, nt = prices.shape

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    oldPos = np.copy(currentPos)

    predict(currentPos, df, 38, list(range(50)), 1, 1.1)

    # Clamp the limit
    for i in range(50):
        clampLimit(currentPos, df, i)

    trackTickerProfit(prices)

    print("Value:", np.sum(tickers__cash + tickers__posVal))
    
    return currentPos


