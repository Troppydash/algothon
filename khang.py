from collections import defaultdict

import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools

from pykalman import KalmanFilter
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM

import matplotlib.pyplot as plt
# UTIL
# CONSTANTS
LIMIT = 10000
INF = 1000000000
COMM_RATE = 0.0010

# GLOBALS
priceMeans = {
    0: 13.845619999999998,
    1: 69.03374000000001,
    2: 47.26258,
    3: 48.00478,
    4: 55.496120000000005,
    5: 11.7407,
    6: 18.177199999999996,
    7: 46.7828,
    8: 68.5373,
    9: 50.185100000000006,
    10: 34.991279999999996,
    11: 26.655479999999997,
    12: 26.612,
    13: 48.495039999999996,
    14: 14.77168,
    15: 25.024019999999997,
    16: 35.25014,
    17: 45.08558,
    18: 13.722539999999999,
    19: 30.998019999999997,
    20: 64.69341999999999,
    21: 22.44104,
    22: 67.5549,
    23: 29.657559999999997,
    24: 63.297619999999995,
    25: 54.63196,
    26: 59.677859999999995,
    27: 28.912860000000002,
    28: 51.157979999999995,
    29: 31.24322,
    30: 13.47922,
    31: 66.98914,
    32: 54.52396000000001,
    33: 42.46486,
    34: 24.871520000000004,
    35: 67.15796,
    36: 36.1705,
    37: 38.82432,
    38: 21.555519999999998,
    39: 49.21754,
    40: 31.651400000000002,
    41: 63.25164000000001,
    42: 12.502360000000001,
    43: 60.843540000000004,
    44: 36.52358,
    45: 52.46146,
    46: 57.53716,
    47: 34.42736,
    48: 41.27304,
    49: 56.123619999999995
}

priceStds = {
    0: 0.7487449857644556,
    1: 3.9081870047027505,
    2: 0.6144378009387482,
    3: 2.0514944334364746,
    4: 1.7339159942021123,
    5: 0.6358842721873459,
    6: 0.2997710549248235,
    7: 1.266560108726797,
    8: 0.5858430969709701,
    9: 3.207362176971684,
    10: 3.131638593797874,
    11: 2.4741732834369032,
    12: 0.5680646044153597,
    13: 1.9269707394928768,
    14: 0.8162026409551763,
    15: 1.1686007423367402,
    16: 0.4320536089909262,
    17: 0.46156640984955,
    18: 0.6975581144579226,
    19: 2.3644529081183703,
    20: 5.786327996836503,
    21: 0.5517758889833141,
    22: 2.474040570299318,
    23: 1.5613213298325173,
    24: 0.8466383694691109,
    25: 9.948234138949823,
    26: 4.129548445184079,
    27: 0.4951840414126242,
    28: 0.38532930975194174,
    29: 0.7848683126080914,
    30: 0.76818031319943,
    31: 2.2172782653993672,
    32: 2.1043823722739825,
    33: 2.651297897536783,
    34: 1.8630766486641377,
    35: 10.17399521307098,
    36: 2.9614391423068573,
    37: 1.8938973128613195,
    38: 0.2559236672269562,
    39: 0.231601872303548,
    40: 0.771222671358691,
    41: 7.833634211547067,
    42: 0.9402253335001727,
    43: 1.898022087355196,
    44: 1.5619817404550635,
    45: 0.7562138501603557,
    46: 4.548283819839261,
    47: 0.8781686573668883,
    48: 3.8149081518188126,
    49: 1.1780585467110527
}

currentPos = np.zeros(50)
oldPos = np.zeros(50)


# UTIL FUNCITONS
def setVolume(newVolume, price):
    if newVolume > 0:
        return int(min(LIMIT // price, newVolume))
    else:
        return int(max(-LIMIT // price, newVolume))


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


def safetyCheck(currentPos, prices, df, ticker, priceMean, priceStd):
    # If the current mean is outside the +2.5std extreme,
    # training data is not reflective of test data. Set position to 0 and stop.
    if (len(prices[ticker]) >= 100):
        upper = priceMean + 2.5 * priceStd
        lower = priceMean - 2.5 * priceStd
        currentMean = sum(prices[ticker][-100:]) / 100

        if not (lower < currentMean < upper):
            currentPos[ticker] = 0

    # Check for limit
    clampLimit(currentPos, df, ticker)


# PAIR TRADING FUNCTIONS
SCALER = 2


def getUnitTradeVolume(prices, share1, share2, trade1, trade2):
    price1 = prices[share1][-1]
    price2 = prices[share2][-1]

    # Get available volume
    vol1 = LIMIT / price1
    vol2 = LIMIT / price2

    unitTrade = min(vol1 / trade1, vol2 / trade2)
    return int(unitTrade * SCALER)


better_pair__last = defaultdict(lambda: 0)
better_pair__amount = defaultdict(lambda: (None, 0))


def better_pair(prices, share1, share2, beta, lower, middle, upper, trade1, trade2, scaler=1.0):
    global better_pair__last, better_pair__amount

    delta = prices[share1][-1] - beta * prices[share2][-1]
    unitTrade = getUnitTradeVolume(prices, share1, share2, trade1, trade2)
    volume1 = int(unitTrade * trade1 * 1)
    volume2 = int(unitTrade * trade2 * 1)

    if delta < lower:
        # print("Buy ", share1, "Sell ", share2)
        better_pair__last[share1, share2] = -1
        better_pair__amount[f"{share1}-{share2}-1"] = (share1, volume1)
        better_pair__amount[f"{share1}-{share2}-2"] = (share2, -volume2)

    if better_pair__last[share1, share2] < 0 and delta >= middle or better_pair__last[
        share1, share2] > 0 and delta <= middle:
        # print("Half the amount of both share")
        better_pair__last[share1, share2] = 0
        better_pair__amount[f"{share1}-{share2}-1"] = (
            share1, int(better_pair__amount[f"{share1}-{share2}-1"][1] // 1.5))
        better_pair__amount[f"{share1}-{share2}-2"] = (
            share2, int(better_pair__amount[f"{share1}-{share2}-2"][1] // 1.5))

    if delta > upper:
        # print("Sell", share1, "Buy", share2)
        better_pair__last[share1, share2] = 1
        better_pair__amount[f"{share1}-{share2}-1"] = (share1, -volume1)
        better_pair__amount[f"{share1}-{share2}-2"] = (share2, volume2)


def better_pair_aggregate(currentPos):
    for key, value in better_pair__amount.items():
        currentPos[value[0]] = 0
    for key, value in better_pair__amount.items():
        currentPos[value[0]] += value[1]
    return currentPos


# SINGLE-SERIES FUNCTION
preTrends = defaultdict(lambda: 0)


def init_movingAvg():
    global preTrends
    preTrends.clear()


def movingAvg(currentPos, prices, ticker: int, priceMean, priceStd, longPeriod=30, shortPeriod=15,
              threshold=0.08):
    global preTrends

    # Check for extremes (outside 1.5 std)
    upper = priceMean + 1.5 * priceStd
    if (prices[ticker][-1] > upper):
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
        preTrends[ticker] = 0
        return
    elif (prices[ticker][-1] < -upper):
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
        preTrends[ticker] = 0
        return

    # Find trend signal by checking moving average cross-over
    if (len(prices[ticker]) <= longPeriod):
        return

    mavgLong = sum(prices[ticker][-longPeriod:]) / longPeriod
    mavgShort = sum(prices[ticker][-shortPeriod:]) / shortPeriod
    diff = mavgShort - mavgLong

    trend = preTrends[ticker]

    # print(diff, trend, preTrends[ticker])

    if diff > threshold:
        trend = 1
    elif diff < -threshold:
        trend = -1

    # print(mavgShort, mavgLong, diff)
    # print(trend, preTrends[ticker])
    # print(prices[ticker][-1])

    # Buy/sell everything
    # when there is a change in trend
    if (trend == 1 and preTrends[ticker] in (-1, 0)):
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
    elif (trend == -1 and preTrends[ticker] in (1, 0)):
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
    # Update preTrends[ticker]
    preTrends[ticker] = trend

    # print("Position: ", currentPos[ticker])
    safe_moving_average(currentPos, ticker)


def safe_moving_average(currentPos, ticker):
    safe_mean_trade(currentPos, [ticker])    


# Very simple mean reversion
# PL: 2, std: 14.97, score: 0.47
def meanRevertGradual(currentPos, prices, ticker: int, priceMean, priceStd):
    diff = prices[ticker][-1] - priceMean
    upper = 1.5 * priceStd
    mid = upper / 2

    limitVol = 2 * LIMIT / prices[ticker][-1]

    # Sell when above mean
    if diff >= upper:
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
        return
    elif diff >= mid:
        currentPos[ticker] = setVolume(currentPos[ticker] - limitVol * (diff - mid) / (upper - mid),
                                       prices[ticker][-1])
        return

    # Buy when below mean
    if diff <= -upper:
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
        return
    elif diff <= -mid:
        currentPos[ticker] = setVolume(currentPos[ticker] + limitVol * (-diff - mid) / (upper - mid),
                                       prices[ticker][-1])
        return


# Flip instantly for quick money when outside of 1 std.
# Might not trade at all, only used for awkward ticker
def meanRevertStrict(currentPos, prices, ticker: int, priceMean, priceStd, upperScale=1):
    diff = prices[ticker][-1] - priceMean
    upper = upperScale * priceStd

    # Sell when above mean 1 std
    if diff >= upper:
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
        return

    # Buy when below mean 1 std
    if diff <= -upper:
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
        return


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


##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

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


direction_mean = defaultdict(lambda: 0)


def mean_trade(df, tickers, beta):
    global currentPos
    spread = np.log(df.iloc[-200:])[tickers] @ beta
    normalized = spread.values[-1] - np.mean(spread)

    threshold = test_threshold(spread)

    # beta adjusted amount

    unit = min(10000 / df[t].values[-1] / abs(b) for t, b in zip(tickers, beta))

    if normalized < -threshold:
        direction_mean[tuple(tickers)] = 1
        # buy
        for i, t in enumerate(tickers):
            currentPos[t] = int(beta[i] * unit)

    elif normalized > threshold:
        direction_mean[tuple(tickers)] = -1

        # sell
        for i, t in enumerate(tickers):
            currentPos[t] = -int(beta[i] * unit)

    elif normalized < 0 and direction_mean[tuple(tickers)] == -1 or normalized > 0 and direction_mean[
        tuple(tickers)] == 1:
        direction_mean[tuple(tickers)] = 0
        for i, t in enumerate(tickers):
            currentPos[t] = 0

    safe_mean_trade(currentPos, tickers)


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

# Return (a, b) for y(t2) = ax(t1) + b
def linearReg(df, t1, t2):
    df = np.log(df)

    x = df[t1]
    x_const = sm.add_constant(x)
    y = df[t2]
    linearModel = sm.OLS(y, x_const)
    result = linearModel.fit()
    return (result.params.values[0], result.params.values[1])

# Use kalman filter for regression
def KalmanFilterRegression(df, t1, t2):
    obs_mat = sm.add_constant(df[t1].values, prepend=False)[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        initial_state_mean=np.ones(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=0.2,
        transition_covariance=0.01 * np.eye(2)
        # em_vars=['observation_covariance', 'transition_covariance']
    )
    state_means, state_covs = kf.filter(df[t2].values)
    return state_means

direction = defaultdict(lambda: 0)
spreads = []
thresholds = []
closes = []
buys = []
sells = []
def pair_trade(df, t1, t2, beta, start=20, fixed_threshold=1.5, period=200, fixed_mean=None, fixed_var=None, 
               convert_rate=-1/2, rolling_beta = False):
    global currentPos

    # Start after 200 points if not using fixed mean/ variance
    if fixed_mean is None and len(df[t1]) < period:
        spreads.append(0)
        thresholds.append(0)
        return
    
    # Try rolling beta
    intercept = 0
    if rolling_beta:
        # Use vecm model - Doesn't work at all
        # vecm_result = VECM(np.log(df[[t1, t2]][-period:]), k_ar_diff=0, coint_rank=1, deterministic='c').fit()
        # beta = vecm_result.beta
        # beta = [beta[0][0], beta[1][0]]

        # Using the cointegration coeff - Dip around 300 for (28, 49) and not too stable
        # Decent for 14-18 though
        jres = get_johansen(np.log(df[[t1, t2]][-period*2:]), 1)
        beta = jres.evecr[:, 0]
        beta = beta/ np.sum(abs(beta))
        # print(beta)

        # Using rolling linear regression - Not working as well as the rolling coint coeff
        # intercept, grad = linearReg(df.iloc[-400:], t1, t2)
        # beta = [-grad, 1]

        # Using kalman filter for the slope - Failed
        # t2 = slope * t1 + intercept
        # result = KalmanFilterRegression(np.log(df[[t1, t2]][-period:]), t1, t2)
        # slope, intercept = result[-1, :]
        # beta = np.array([-slope, 1])
        # beta = beta/ np.sum(abs(beta))
        # print(beta)

        
    
    # spread = np.log(df.iloc[-period:])[[t1, t2]] @ beta
    # Remove the log
    spread = df.iloc[-period:][[t1, t2]] @ beta
    normalized = spread - intercept
    if fixed_mean is None:
        normalized = (normalized - np.mean(normalized))/np.std(normalized)
    else:
        normalized = (normalized - fixed_mean)/fixed_var
    spreads.append(normalized.values[-1])

    # NORMALISATION WITH KALMAN FILTER - Failed
    # # Use kalman filter to find the normalized spread
    # kalman_filter = KalmanFilter(transition_matrices = [1],
    #                     observation_matrices = [1],
    #                     initial_state_mean = spread.iloc[:start].mean(),
    #                     observation_covariance = spread.iloc[:start].var(),
    #                     em_vars=['transition_covariance', 'initial_state_covariance'])
    # kalman_filter = kalman_filter.em(spread.iloc[:start], n_iter=5)
    # (filtered_state_means, filtered_state_covariances) = kalman_filter.filter(spread.iloc[:start])
    
    # # Obtain the current Mean and Covariance Matrix expectations.
    # current_mean = filtered_state_means[-1, :]
    # current_cov = filtered_state_covariances[-1, :]
    
    # # Initialize a mean series for spread normalization using the Kalman Filter's results.
    # mean_series = np.array([None]*(spread.shape[0]-start))
    
    # # Roll over the Kalman Filter to obtain the mean series.
    # for i in range(start, spread.shape[0]):
    #     (current_mean, current_cov) = kalman_filter.filter_update(filtered_state_mean = current_mean,
    #                                                             filtered_state_covariance = current_cov,
    #                                                             observation = spread.iloc[i])
    #     mean_series[i-start] = float(current_mean)

    # # Obtain the normalized spread series.
    # normalized = (spread.iloc[start:] - mean_series)
    

    if fixed_var is not None:
        threshold = fixed_threshold
    else:
        threshold = test_threshold(normalized)
    thresholds.append(threshold)
    
    unit = min(10000 / df[t].values[-1] / abs(b) for t, b in zip([t1, t2], beta))

    if normalized.values[-1] < -threshold and direction[(t1, t2)] != 1:
        direction[(t1, t2)] = 1
        # buy
        currentPos[t1] = int(beta[0] * unit)
        currentPos[t2] = int(beta[1] * unit)
        buys.append(len(df[t1]))

    elif normalized.values[-1] > threshold and direction[(t1, t2)] != -1:
        direction[(t1, t2)] = -1

        # sell
        currentPos[t1] = -int(beta[0] * unit)
        currentPos[t2] = -int(beta[1] * unit)
        sells.append(len(df[t1]))

    elif normalized.values[-1] < threshold * convert_rate and direction[(t1, t2)] == -1 or \
            normalized.values[-1] > -threshold * convert_rate and direction[(t1, t2)] == 1:
        closes.append(len(df[t1]))
        direction[(t1, t2)] = 0
        currentPos[t1] = currentPos[t2] = 0

    safe_pair_trade(currentPos, t1, t2)
    print(closes)
    print(buys)
    print(sells)


# Safety: Turn off the pair trade if PnL is below -1k
def safe_pair_trade(currentPos, t1, t2):
    safe_mean_trade(currentPos, [t1, t2])


def getMyPosition(prices):
    global currentPos, oldPos

    nins, nt = prices.shape

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    oldPos = np.copy(currentPos)

    if False:
        # better_pair(prices, 28, 39, 1.2159226859939878, -8.963364425168958, -8.783364425168958,
        #             -8.603364425168959, 7.0, 8.0, 0.6)
        better_pair(prices, 14, 30, 0.7192064351085297, 4.86640231, 5.15640231, 5.44640231, 6, 4)
        # better_pair(prices, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299,
        #             39.807363604822996,
        #             12.0, 2.0, 0.8)
        currentPos = better_pair_aggregate(currentPos)

    if True:
        # Doesn't work since it doesn't mean revert in 500-750
        # using
        # https://www.quantconnect.com/docs/v2/research-environment/applying-research/kalman-filters-and-stat-arb
        # and
        # https://www.quantconnect.com/docs/v2/research-environment/applying-research/pca-and-pairs-trading

        # pair_trade(df, 11, 42, [0.45263609, -0.54736391], 0.026805109292277557)
        # pair_trade(df, 1, 10, [0.63057265, -0.36942735], 0.019607962852651883)
        # pair_trade(df, 4, 32, [0.4989909853629603, -0.5010090146370396], 0.008144572661827587)
        # pair_trade(df, 24, 49, [0.4927626542556702, -0.5072373457443298])
        # pair_trade(df, 22, 47, [0.4570128708881386, -0.5429871291118614])

        # 14-18 works for 500 - 750, but not sure if this continues
        # pair_trade(df, 14, 18, [1.000000, -0.814115], rolling_beta=True)

        # Try other pairs:
        # (28, 49): Failed
        # (36, 42): Failed (Positive PnL, not stable)
        # (43, 46): Failed
        # (43, 49): Failed
        # (20, 35): Failed for rolling, no rolling still has initial negative (but positive PnL)
        # (14, 36): Pretty decent, but 14 is already paired with 18

        pair_trade(df, 24, 49, [1, -1.76],  
                   fixed_mean=-35.8631276, fixed_var=1.4104603201076735,
                   convert_rate=1/4, 
                   start=40, period=200, rolling_beta=False)
        
        # pair_trade(df, 3, 34, [1, -0.53],  
        #            fixed_mean=2.170022142435276, fixed_var=0.01215140613320954,
        #            convert_rate=-1/3, 
        #            start=40, period=200, rolling_beta=False)
        pass

    if False:
        mean_trade(df, [15, 16, 38], [0.1322021733431518, 0.5850307797427331, -0.2827670469141151])
        mean_trade(df, [9, 21, 35], [0.32971776797284735, -0.5696967968104495, -0.10058543521670305])
        mean_trade(df, [7, 17, 25], [0.38177208101532123, 0.5859183559138036, 0.03230956307087521])
        mean_trade(df, [27, 40, 44], [0.5904390715664551, -0.30280905334000785, -0.10675187509353694])

    if False:
        # # LEAD LAG TRADE:
        predict(currentPos, df, 38, list(range(50)), 1, 1.1)
        # predict(currentPos, df, 27, list(range(50)), 1, 1.1)

        # # SINGLE TRADE:
        # # SAFE TICKERS: Gain positive PL and score on themselves and overall
        # # For ticker 8, simple mean reversion (TODO: See if there is a better way)
        # # Increase performance by .1
        # meanRevertGradual(currentPos, prices, 8, 68.537300, 0.585843)

        # # Ticker 27: Slow, significant trend. Increase performance by .2
        # movingAvg(currentPos, prices, 27, 28.912860, 0.495184, 40, 20, threshold=0.05)
        
        # # Ticker 3: Moving average
        # movingAvg(currentPos, prices, 3, 48.004780, 2.051494, 40, 20, threshold=0.2)
        
        # All of those hurts performance from 500 - 750
        # # Ticker 6: Confirmed stationary-ish with AD-fuller at 10% sig level
        # # Less risky, higher PL, but with higher Std, so lower score.
        # # Score is still positive, but around 40% of the score is negative
        # meanRevertStrict(currentPos, prices, 6, 18.177200, 0.299771)
        
        # RISKY STICKER: Gain positive PL, but std makes negative score individually.
        # Group them into groups => Diversification = Overall score improvements
        
        # # Ticker 4: Mean reversion (TODO: other strategy needs more careful investigation)
        # # Mean revert: Makes decent PL, but std varies due to the fluctuation of the price
        # meanRevertStrict(currentPos, prices, 4, 55.496120, 1.733916, 1.5)
        
        # # Ticker 15: Too jaggy trend to fit ARIMA. Try using moving average for trend prediction.
        # # Can't do fair price with moving average on short window.
        # # Work for moving average with short window and low threshold due to the short trend cycle.
        # # Quite high std (due to the price itself is volatile), but high PL => Score improvement
        # # with diversification
        # movingAvg(currentPos, prices, 15, 25.024019999999997, 1.1686007423367402, 20, 10, 0)

    # Run safety check on all tickers
    for i in [8,6]:
        safetyCheck(currentPos, prices, df, i, priceMeans[i], priceStds[i])

    # Clamp the limit
    for i in range(50):
        clampLimit(currentPos, df, i)

    trackTickerProfit(prices)

    return currentPos

def funcGen(currentPair: tuple):
    def getMyPosition(prices):
        global currentPos, oldPos

        nins, nt = prices.shape

        df = pd.DataFrame(prices.T, columns=np.arange(50))

        oldPos = np.copy(currentPos)

        if False:
            # better_pair(prices, 28, 39, 1.2159226859939878, -8.963364425168958, -8.783364425168958,
            #             -8.603364425168959, 7.0, 8.0, 0.6)
            better_pair(prices, 14, 30, 0.7192064351085297, 4.86640231, 5.15640231, 5.44640231, 6, 4)
            # better_pair(prices, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299,
            #             39.807363604822996,
            #             12.0, 2.0, 0.8)
            currentPos = better_pair_aggregate(currentPos)

        if True:
            # Doesn't work since it doesn't mean revert in 500-750
            # using
            # https://www.quantconnect.com/docs/v2/research-environment/applying-research/kalman-filters-and-stat-arb
            # and
            # https://www.quantconnect.com/docs/v2/research-environment/applying-research/pca-and-pairs-trading

            # pair_trade(df, 11, 42, [0.45263609, -0.54736391], 0.026805109292277557)
            # pair_trade(df, 1, 10, [0.63057265, -0.36942735], 0.019607962852651883)
            # pair_trade(df, 4, 32, [0.4989909853629603, -0.5010090146370396], 0.008144572661827587)
            # pair_trade(df, 24, 49, [0.4927626542556702, -0.5072373457443298])
            # pair_trade(df, 22, 47, [0.4570128708881386, -0.5429871291118614])

            # 14-18 works for 500 - 750, but not sure if this continues
            # pair_trade(df, 14, 18, [1.000000, -0.814115], rolling_beta=True)

            # Try other pairs:
            # (28, 49): Failed
            # (36, 42): Failed (Positive PnL, not stable)
            # (43, 46): Failed
            # (43, 49): Failed
            # (20, 35): Failed for rolling, no rolling still has initial negative (but positive PnL)
            # (14, 36): Pretty decent, but 14 is already paired with 18
            pair_trade(df, currentPair[0], currentPair[1], [0, 0], start=40, period=200, rolling_beta=True)

        if False:
            mean_trade(df, [15, 16, 38], [0.1322021733431518, 0.5850307797427331, -0.2827670469141151])
            mean_trade(df, [9, 21, 35], [0.32971776797284735, -0.5696967968104495, -0.10058543521670305])
            mean_trade(df, [7, 17, 25], [0.38177208101532123, 0.5859183559138036, 0.03230956307087521])
            mean_trade(df, [27, 40, 44], [0.5904390715664551, -0.30280905334000785, -0.10675187509353694])

        if False:
            # # LEAD LAG TRADE:
            predict(currentPos, df, 38, list(range(50)), 1, 1.1)
            predict(currentPos, df, 27, list(range(50)), 1, 1.1)

            # # SINGLE TRADE:
            # # SAFE TICKERS: Gain positive PL and score on themselves and overall
            # # For ticker 8, simple mean reversion (TODO: See if there is a better way)
            # # Increase performance by .1
            # meanRevertGradual(currentPos, prices, 8, 68.537300, 0.585843)

            # # Ticker 27: Slow, significant trend. Increase performance by .2
            # movingAvg(currentPos, prices, 27, 28.912860, 0.495184, 40, 20, threshold=0.05)
            
            # # Ticker 3: Moving average
            # movingAvg(currentPos, prices, 3, 48.004780, 2.051494, 40, 20, threshold=0.2)
            
            # All of those hurts performance from 500 - 750
            # # Ticker 6: Confirmed stationary-ish with AD-fuller at 10% sig level
            # # Less risky, higher PL, but with higher Std, so lower score.
            # # Score is still positive, but around 40% of the score is negative
            # meanRevertStrict(currentPos, prices, 6, 18.177200, 0.299771)
            
            # RISKY STICKER: Gain positive PL, but std makes negative score individually.
            # Group them into groups => Diversification = Overall score improvements
            
            # # Ticker 4: Mean reversion (TODO: other strategy needs more careful investigation)
            # # Mean revert: Makes decent PL, but std varies due to the fluctuation of the price
            # meanRevertStrict(currentPos, prices, 4, 55.496120, 1.733916, 1.5)
            
            # # Ticker 15: Too jaggy trend to fit ARIMA. Try using moving average for trend prediction.
            # # Can't do fair price with moving average on short window.
            # # Work for moving average with short window and low threshold due to the short trend cycle.
            # # Quite high std (due to the price itself is volatile), but high PL => Score improvement
            # # with diversification
            # movingAvg(currentPos, prices, 15, 25.024019999999997, 1.1686007423367402, 20, 10, 0)

        # Run safety check on all tickers
        for i in [8,6]:
            safetyCheck(currentPos, prices, df, i, priceMeans[i], priceStds[i])

        # Clamp the limit
        for i in range(50):
            clampLimit(currentPos, df, i)

        trackTickerProfit(prices)

        return currentPos
    return getMyPosition

if __name__ == "__main__":
    from custom_eval.alleval import all_eval, prcAll

    pairs = []
    for i in range(14, 50):
        for j in range(i+1, 50):
            currentPair = (i, j)
            print(i, j)

            if prcAll is not None:
                testPrices = prcAll[:, 300:501]
                df = pd.DataFrame(testPrices.T, columns=np.arange(50))
                testCointPval = stattools.coint(df[i], df[j])[1]
                if testCointPval > 0.1:
                    continue


            result = (all_eval(currentPos, funcGen(currentPair), 1, talkative=False))
            if result[0] > 1 and result[1] > 0:
                pairs.append((currentPair, result))
                print(pairs)
    print(pairs)
