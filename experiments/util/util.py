from util.constants import LIMIT

def setVolume(newVolume, price):
    if newVolume > 0:
        return int(min(LIMIT//price, newVolume))
    else:
        return int(max(-LIMIT//price, newVolume))

def safetyCheck(currentPos, prices, ticker, priceMean, priceStd):
    # If the current mean is outside the +2std extreme, 
    # training data is not reflective of test data. Set position to 0 and stop.
    if (len(prices[ticker]) >= 100):
        upper = priceMean + 2 * priceStd
        currentMean = sum(prices[ticker][-100:])/ 100

        if not(-upper < currentMean < upper):
            currentPos[ticker] = 0