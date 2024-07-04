from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.tsa.ardl as ardl

# UTIL
# CONSTANTS
LIMIT = 10000
INF = 1000000000
COMM_RATE = 0.0010


# UTIL FUNCITONS
def setVolume(newVolume, price):
    if newVolume > 0:
        return int(min(LIMIT // price, newVolume))
    else:
        return int(max(-LIMIT // price, newVolume))


def safetyCheck(currentPos, prices, df, ticker, priceMean, priceStd):
    # If the current mean is outside the +2.5std extreme, 
    # training data is not reflective of test data. Set position to 0 and stop.
    if (len(prices[ticker]) >= 100):
        upper = priceMean + 2.5 * priceStd
        lower = priceMean - 2.5 * priceStd
        currentMean = sum(prices[ticker][-100:]) / 100

        if not (lower < currentMean < upper):
            currentPos[ticker] = 0

    # check for limits
    i = ticker
    pos, val = currentPos[i], df[i].values[-1]
    if pos * val > 10000:
        currentPos[i] = int(10000 / val)
    if pos * val < -10000:
        currentPos[i] = int(-10000 / val)


# PAIR TRADING FUNCTIONS
SCALER = 2


def getUnitTradeVolume(prices, share1, share2, trade1, trade2):
    price1 = prices[share1][-1]
    price2 = prices[share2][-1]

    # Get available volume
    vol1 = LIMIT / price1
    vol2 = LIMIT / price2

    unitTrade = min(vol1 / trade1, vol2 / trade2)
    return int(unitTrade*SCALER)


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


def getMyPosition(prices):
    global currentPos
    nins, nt = prices.shape
    if nt < 2:
        return np.zeros(nins)

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    # PAIR TRADING: Terry's part. FAILED for 500-750 for some reason.

    # # ok
    # better_pair(prices, 12, 34, 0.23829431834048995, 20.657766061851223, 20.757766061851225,
    #             20.857766061851226, 9.0, 2.0, 0.6)
    # better_pair(prices, 12, 25, -0.06366267571277003, 29.863092873692395, 29.983092873692396,
    #             30.103092873692397, 16.0, 1.0, 0.6)

    # better_pair(prices, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299, 39.807363604822996,
    #             12.0, 2.0, 0.8)
    # better_pair(prices, 7, 19, 0.3973201339866572, 33.5724356, 34.1524356, 34.7324356, 13, 5, 0.3)

    # better_pair(prices, 28, 39, 1.2159226859939878, -8.963364425168958, -8.783364425168958,
    #             -8.603364425168959, 7.0, 8.0, 0.6)
    # better_pair(prices, 28, 41, 0.021148, 49.6408878, 49.8208878, 50.0008878, 48, 1, 0.6)


    # better_pair(prices, 2, 11, 0.22442595136426546, 41.24598604371131, 41.35598604371131, 41.46598604371131,
    #             9.0, 2.0, 2)
    # better_pair(prices, 11, 42, 2.81, -9.48586161, -8.89586161, -8.30586161, 8, 21, 0.6)

    # better_pair(prices, 43, 49, 0.2046650849773665, 47.71489042555398, 48.64489042555398, 49.57489042555398,
    #             10, 2)

    # # Caution pair
    # better_pair(prices, 14, 30, 0.7192064351085297, 4.86640231, 5.15640231, 5.44640231, 6, 4)
    # # Risky pair
    # better_pair(prices, 22, 24, 3.4117759223360125, -149.87702011, -148.6872011, -147.49702011, 8, 27)

    # # New pairs added: Improve the performance across min, mean, max, but also widen the std (so takes caution, prob re-check?)
    # currentPos = better_pair_aggregate(currentPos)

    # LEAD LAG TRADE:
    predict(currentPos, df, 38, list(range(50)), 1, 1.1)
    predict(currentPos, df, 27, list(range(50)), 1, 1.1)

    # SINGLE TRADE:
    # SAFE TICKERS: Gain positive PL and score on themselves and overall
    # For ticker 8, simple mean reversion (TODO: See if there is a better way)
    # Increase performance by .1
    meanRevertGradual(currentPos, prices, 8, 68.537300, 0.585843)

    # Ticker 27: Slow, significant trend. Increase performance by .2
    # movingAvg(currentPos, prices, 27, 28.912860, 0.495184, 40, 20, threshold=0.1)

    # Ticker 3: Moving average
    movingAvg(currentPos, prices, 3, 48.004780, 2.051494, 40, 20, threshold=0.2)

    # # Ticker 6: Confirmed stationary-ish with AD-fuller at 10% sig level
    # # Less risky, higher PL, but with higher Std, so lower score.
    # # Score is still positive, but around 40% of the score is negative
    # meanRevertStrict(currentPos, prices, 6, 18.177200, 0.299771)

    # RISKY STICKER: Gain positive PL, but std makes negative score individually.
    # Group them into groups => Diversification = Overall score improvements

    # Ticker 4: Mean reversion (TODO: other strategy needs more careful investigation)
    # Mean revert: Makes decent PL, but std varies due to the fluctuation of the price
    # meanRevertStrict(currentPos, prices, 4, 55.496120, 1.733916, 1.5)

    # Ticker 15: Too jaggy trend to fit ARIMA. Try using moving average for trend prediction.
    # Can't do fair price with moving average on short window.
    # Work for moving average with short window and low threshold due to the short trend cycle.
    # Quite high std (due to the price itself is volatile), but high PL => Score improvement
    # with diversification
    # movingAvg(currentPos, prices, 15, 25.024019999999997, 1.1686007423367402, 20, 10, 0)

    # Run safety check on all tickers
    for i in [8, 6, 4]:
        safetyCheck(currentPos, prices, df, i, priceMeans[i], priceStds[i])

    return currentPos
