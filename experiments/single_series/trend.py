import numpy as np
from collections import defaultdict

from util.constants import INF, LIMIT, COMM_RATE
from util.util import setVolume

# Dev for ticker 15: Doesn't work since the moving average is either lagging
# or unstable
movingAverageTrend__movingAvg = defaultdict(lambda: [])
isPositiveInf = False
def initMovingAvergaeTrend():
    global isPositiveInf
    movingAverageTrend__movingAvg.clear()
    isPositiveInf = False

def movingAverageTrend(currentPos, prices, ticker, period=20):
    global isPositiveInf
    if len(prices[ticker]) < period:
        return
    
    currentMovingAvg = sum(prices[ticker][-period:]) / period
    allMovingAvg = movingAverageTrend__movingAvg[ticker]
    allMovingAvg.append(currentMovingAvg)

    if len(allMovingAvg) < 3:
        return
    
    grad, intercept = np.polyfit(range(1, 4), allMovingAvg[-3:], 1)
    print(grad)
    if grad > 0.01 and not isPositiveInf:
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])
        isPositiveInf = True

        print("BUY")
        print(currentMovingAvg)
        print(grad, currentPos[ticker])
        print(prices[ticker][-1])

    elif grad < -0.01 and isPositiveInf:
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
        isPositiveInf = False

        print("SELL")
        print(currentMovingAvg)
        print(grad, currentPos[ticker])
        print(prices[ticker][-1])
    