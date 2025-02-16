from util.util import setVolume
from util.constants import LIMIT, INF
from collections import defaultdict

# Made for Ticker 8: Doesn't work (due to large short-term fluctuations)
linreg__buyGap = 0.01
linreg__sellGap = 0.01
def linreg(currentPos, prices, predictError, model, ticker: int):
    # Load the model with: model = sm.load("...") for trained arima model
    if (len(prices[ticker]) <= 20):
        return

    applied = model.apply(prices[ticker][-20:])
    nextPrice = applied.forecast(1)[0]

    buyPrice = nextPrice - linreg__buyGap - predictError
    sellPrice = nextPrice + linreg__sellGap + predictError

    # print(nextPrice, buyPrice, sellPrice)
    # print(prices[ticker][-1])

    limitVol = LIMIT/prices[ticker][-1]

    if prices[ticker][-1] > sellPrice:
        currentPos[ticker] = setVolume(-INF, prices[ticker][-1])
    elif prices[ticker][-1] < buyPrice:
        currentPos[ticker] = setVolume(INF, prices[ticker][-1])