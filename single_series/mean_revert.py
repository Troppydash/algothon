from util.constants import INF, LIMIT
from util.util import setVolume
from collections import defaultdict

# Try moving average

preTrends = defaultdict(lambda: 0)
def init_movingAvg():
    global preTrends
    preTrends.clear()

def movingAvg(currentPos, prices, ticker: int, priceMean, priceStd, longPeriod = 30, shortPeriod = 15, 
              threshold = 0.08):
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
    mid = upper/2

    limitVol = 2 * LIMIT/prices[ticker][-1]
    
    # Sell when above mean
    if diff >= upper:
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
        return
    elif diff >= mid:
        currentPos[ticker] = setVolume(currentPos[ticker] - limitVol * (diff - mid)/(upper - mid), prices[ticker][-1])
        return
    
    # Buy when below mean
    if diff <= -upper:
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
        return
    elif diff <= -mid:
        currentPos[ticker] = setVolume(currentPos[ticker] + limitVol * (-diff - mid)/(upper - mid), prices[ticker][-1])
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