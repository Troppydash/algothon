from collections import defaultdict
import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl
from statsmodels.tsa.vector_ar.vecm import coint_johansen

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

# PAIR TRADE
# Safety: Turn off the mean trade if PnL is below -1k for the group of tickers
safe_mean_trade__fail = defaultdict(lambda: False)


def safe_mean_trade(currentPos, tickers):
    global tickers__cash, tickers__posVal
    tickers = tuple(tickers)
    # If got below 1.5k, something is wrong with the strategy. Turns it off.
    if (safe_mean_trade__fail[tickers] == True):
        for ticker in tickers:
            currentPos[ticker] = 0
        return

    # Check if got below -1000
    SAFETY_THRESHOLD = -1000
    totalPL = 0
    for ticker in tickers:
        totalPL += tickers__cash[ticker] + tickers__posVal[ticker]

    if totalPL < SAFETY_THRESHOLD:
        safe_mean_trade__fail[tickers] = True
        for ticker in tickers:
            currentPos[ticker] = 0
    return

# Safety: Turn off the pair trade if PnL is below -1k
def safe_pair_trade(currentPos, t1, t2):
    safe_mean_trade(currentPos, [t1, t2])

def get_johansen(y, p):
    """
    Get the cointegration vectors at 95% level of significance
    given by the trace statistic test.
    """

    N, l = y.shape
    jres = coint_johansen(y, 0, p)
    trstat = jres.lr1                       # trace statistic
    tsignf = jres.cvt                       # critical values

    if N > 12:
        jres.r = l
        jres.evecr = jres.evec
        return jres

    for i in range(l):
        if trstat[i] > tsignf[i, 1]:     # 0: 90%  1:95% 2: 99%
            r = i + 1
    jres.r = r
    jres.evecr = jres.evec[:, :r]

    return jres

def test_threshold(spread):
    # https://www.quantconnect.com/docs/v2/research-environment/applying-research/kalman-filters-and-stat-arb
    # Basically find the "expected value "
    s0 = np.linspace(0, spread.max(), 50)
    f_bar = np.array([None] * 50)
    for i in range(50):
        f_bar[i] = len(spread.values[spread.values > s0[i]]) / spread.shape[0]

    D = np.zeros((49, 50))
    for i in range(D.shape[0]):
        D[i, i] = 1
        D[i, i + 1] = -1

    l = 1.0
    f_star = np.linalg.inv(np.eye(50) + l * D.T @ D) @ f_bar.reshape(-1, 1)
    s_star = [f_star[i] * s0[i] for i in range(50)]
    threshold = s0[s_star.index(max(s_star))]
    # print(threshold)
    return threshold

direction = defaultdict(lambda: 0)
spreads = []
thresholds = []
closes = []
def pair_trade(df, t1, t2, beta, start=20, threshold=0, period=200, rolling_beta = False):
    global currentPos
    if len(df[t1]) < period:
        spreads.append(0)
        thresholds.append(0)
        return
    
    # Try rolling beta
    intercept = 0
    if rolling_beta:

        # Using the cointegration coeff - Dip around 300 for (28, 49) and not too stable
        # Decent for 14-18 though
        jres = get_johansen(np.log(df[[t1, t2]][-period*2:]), 1)
        beta = jres.evecr[:, 0]
        beta = beta/ np.sum(abs(beta))

        
    
    spread = np.log(df.iloc[-period:])[[t1, t2]] @ beta
    normalized = spread - intercept
    normalized = (normalized - np.mean(normalized))/np.std(normalized)
    spreads.append(normalized.values[-1])

    threshold = test_threshold(normalized)
    thresholds.append(threshold)
    
    unit = min(10000 / df[t].values[-1] / abs(b) for t, b in zip([t1, t2], beta))

    if normalized.values[-1] < -threshold:
        direction[(t1, t2)] = 1
        # buy
        currentPos[t1] = int(beta[0] * unit)
        currentPos[t2] = int(beta[1] * unit)

    elif normalized.values[-1] > threshold:
        direction[(t1, t2)] = -1

        # sell
        currentPos[t1] = -int(beta[0] * unit)
        currentPos[t2] = -int(beta[1] * unit)

    elif normalized.values[-1] < -threshold / 2 and direction[(t1, t2)] == -1 or normalized.values[-1] > threshold / 2 and \
            direction[(t1, t2)] == 1:
        closes.append(len(df[t1]))
        direction[(t1, t2)] = 0
        currentPos[t1] = currentPos[t2] = 0
    safe_pair_trade(currentPos, t1, t2)


pls = []
values = [0]
def getMyPosition(prices):
    global currentPos, oldPos, pls, values

    nins, nt = prices.shape

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    oldPos = np.copy(currentPos)

    # predict(currentPos, df, 38, list(range(50)), 1, 1.1)
    pair_trade(df, 14, 18, [1.000000, -0.814115], rolling_beta=True)

    # Clamp the limit
    for i in range(50):
        clampLimit(currentPos, df, i)

    trackTickerProfit(prices)

    todayVal = np.sum(tickers__cash + tickers__posVal)
    todayPl = todayVal - values[-1]

    pls.append(todayPl)
    values.append(todayVal)

    if len(df[0]) == 750:
        # startIndex = 500
        # endIndex = 699

        # outputStr = ""
        # outputStr += str(df[0][699]) + "|"
        
        # start = df[:].values[startIndex]
        # end = df[:].values[endIndex]
        # diff = end - start

        # for i in range(50):
        #     if diff[i] < 0:
        #         diff[i] += start[i] * 2 * COMM_RATE
        #     else:
        #         diff[i] -= start[i] * 2 * COMM_RATE

        # profit = ""
        # for i in range(50):
        #     if abs(diff[i]) > 4:
        #         profit += f"{i}:{int(diff[i] * 10)/10:.1f};"
        # outputStr += profit
        # raise Exception(outputStr)
        raise Exception(f"{np.mean(pls) - 0.1 * np.std(pls)}/0")

    
    
    return currentPos


