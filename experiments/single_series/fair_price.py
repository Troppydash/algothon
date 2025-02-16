from util.constants import COMM_RATE, INF
from util.util import setVolume

# getFairPrice method
# Moving average on short time frame: Hopefully works on those with high noise,
# but slow/smooth trend.
# Nope, doesn't work, since the rule for fair price doesn't apply on 
# strong up/ down trend for moving average.
# def getMovingAverage(prices, ticker, window=7):
#     # Return INFINITY if not enough values
#     if len(prices[ticker]) < window:
#         return INF
    
#     return sum(prices[ticker][-window:])/window

# Simple strategy: Sell when above fair price, buy when below
# getFairPrice is a method to find the fair price (eg: taking the 
# moving average in a short window)
def fairPriceStrategy(currentPos, prices, ticker, getFairPrice):
    fairPrice = getFairPrice(prices, ticker)
    if fairPrice == INF:
        return
    currentPrice = prices[ticker][-1]

    # Take out the potential commission
    buyPrice = fairPrice - COMM_RATE * currentPrice
    sellPrice = fairPrice + COMM_RATE * currentPrice

    if currentPrice < buyPrice:
        currentPos[ticker] = setVolume(INF, currentPrice)
    elif currentPrice > sellPrice:
        currentPos[ticker] = setVolume(-INF, currentPrice)

    print(currentPrice, buyPrice, fairPrice, sellPrice)
    print(currentPos[ticker])