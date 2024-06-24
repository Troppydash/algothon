from util.constants import INF, LIMIT
from util.util import setVolume

# Try moving average

preTrend = 0
def movingAvg(currentPos, prices, ticker: int, priceMean, priceStd):
    global preTrend

    # Check for extremes (outside 1.5 std)
    upper = priceMean + 1.5 * priceStd
    if (prices[ticker][-1] > upper):
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
        return
    elif (prices[ticker][-1] < -upper):
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
        return

    # Find trend signal by checking moving average cross-over
    if (len(prices[ticker]) <= 40):
        return
    
    mavg40 = sum(prices[ticker][-40:]) / 40
    mavg20 = sum(prices[ticker][-20:]) / 20

    trend = 1 if mavg20 > mavg40 else -1
    
    # If the 2 extremes above doesn't work, then buy/sell everything
    # when there is a change in trend
    if (trend == 1 and preTrend in (-1, 0) and prices[ticker][-1] < priceMean):
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
    elif (trend == -1 and preTrend in (1, 0) and prices[ticker][-1] > priceMean):
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
    # Update preTrend
    preTrend = trend
    


# Very simple mean reversion
# PL: 2, std: 14.97, score: 0.47
def meanRevert(currentPos, prices, ticker: int, priceMean, priceStd):
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
