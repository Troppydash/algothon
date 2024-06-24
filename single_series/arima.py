import statsmodels.api as sm
from util.util import setVolume
from util.util import LIMIT

# Made for Ticker 8: Doesn't work (due to large short-term fluctuations)
def linreg(currentPos, prices, priceMean, priceChangeStd, model, ticker: int):
    # Load the model with: model = sm.load("...") for trained arima model
    if (len(prices[ticker]) <= 4):
        return

    applied = model.apply(prices[ticker][-20:] - priceMean)
    nextPrice = applied.forecast(3)[2] + priceMean

    if nextPrice - prices[ticker][-1] > priceChangeStd:
        currentPos[ticker] = setVolume(-LIMIT, prices[ticker][-1])
    elif nextPrice - prices[ticker][-1] < -priceChangeStd:
        currentPos[ticker] = setVolume(LIMIT, prices[ticker][-1])