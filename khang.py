from collections import defaultdict

import numpy as np
import pandas as pd

import statsmodels.api as sm

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

currentPos = np.zeros(50)
limit = 10000
SCALER = 20
INF = 1000000000
commRate = 0.0010
maxPrices = {
    0 :  15.56 ,
    1 :  75.15 ,
    2 :  48.61 ,
    3 :  51.01 ,
    4 :  59.12 ,
    5 :  13.0 ,
    6 :  18.98 ,
    7 :  51.09 ,
    8 :  69.99 ,
    9 :  59.1 ,
    10 :  40.64 ,
    11 :  33.53 ,
    12 :  27.61 ,
    13 :  51.42 ,
    14 :  16.98 ,
    15 :  27.23 ,
    16 :  36.24 ,
    17 :  46.13 ,
    18 :  14.75 ,
    19 :  35.47 ,
    20 :  73.68 ,
    21 :  23.34 ,
    22 :  72.21 ,
    23 :  32.77 ,
    24 :  64.97 ,
    25 :  71.09 ,
    26 :  69.19 ,
    27 :  29.96 ,
    28 :  52.11 ,
    29 :  32.61 ,
    30 :  15.35 ,
    31 :  70.85 ,
    32 :  59.12 ,
    33 :  48.98 ,
    34 :  28.55 ,
    35 :  82.12 ,
    36 :  43.11 ,
    37 :  42.23 ,
    38 :  22.05 ,
    39 :  49.71 ,
    40 :  33.26 ,
    41 :  77.47 ,
    42 :  14.5 ,
    43 :  66.09 ,
    44 :  39.13 ,
    45 :  54.15 ,
    46 :  66.6 ,
    47 :  36.48 ,
    48 :  50.02 ,
    49 :  59.31
}

def setVolume(newVolume, price):
    global limit
    if newVolume > 0:
        return int(min(limit//price, newVolume))
    else:
        return int(max(-limit//price, newVolume))

def getUnitTradeVolume(prices, share1, share2, trade1, trade2):
    global limit

    price1 = prices[share2][-1]
    price2 = prices[share1][-1]

    # Get available volume
    vol1 = limit/ price1
    vol2 = limit/ price2

    unitTrade = min(vol1/trade1, vol2/trade2)
    return unitTrade * SCALER


last = defaultdict(lambda: 0)
def better_pair(prices, share1, share2, beta, lower, middle, upper, trade1, trade2):
    global last, currentPos
    
    delta = prices[share1][-1] - beta * prices[share2][-1]
    unitTrade = getUnitTradeVolume(prices, share1, share2, trade1, trade2)
    volume1 = unitTrade * trade1
    volume2 = unitTrade * trade2

    if delta < lower and last[share1,share2] != -1:
        last[share1,share2] = -1
        currentPos[share1] = int(volume1)
        currentPos[share2] = -int(volume2)
    
    if last[share1,share2] < 0 and delta > middle or last[share1,share2] > 0 and delta < middle:
        last[share1, share2] = 0
        currentPos[share1] //= 3
        currentPos[share2] //= 3

    if delta > upper and last[share1,share2] != 1:
        last[share1,share2] = 1
        currentPos[share1] = -int(volume1)
        currentPos[share2] = int(volume2)


# Ticker 8
# Doesn't work for some reason

error = 0
prePred = -10
first = True
model8 = None

def linreg8(prices):
    if (len(prices[8]) <= 4):
        return
    
    global currentPos, error, prePred, first

    if (model8 == None):
        model8 = sm.load("model8.pickle")

    price8 = prices[8]

    priceMean = 68.537300
    priceChangeMean = -0.0008617234468937727
    priceChangeStd = 0.10007741951924473

    applied8 = model8.apply(price8[-20:] - priceMean)
    nextPrice8 = applied8.forecast(3)[2] + priceMean
    
    print(nextPrice8)
    print(price8[-1])

    if nextPrice8 - price8[-1] > 0.01:
        currentPos[8] = setVolume(-limit, price8[-1])
    elif nextPrice8 - price8[-1] < -0.01:
        currentPos[8] = setVolume(limit, price8[-1])

# Try moving average
preTrend = 0
def movingAvg(prices):
    global currentPos, limit, preTrend

    # print(currentPos[8])

    price8 = prices[8]
    priceStd = 0.585843
    priceMean = 68.537300

    # Check for extremes (outside 1.5 std)
    upper = priceMean + 1.5 * priceStd
    if (price8[-1] > upper):
        currentPos[8] = setVolume(-INF, price8[-1])
        return
    elif (price8[-1] < -upper):
        currentPos[8] = setVolume(INF, price8[-1])
        return

    # Find trend signal by checking moving average cross-over
    if (len(prices[8]) <= 40):
        return
    mavg40 = sum(price8[-40:]) / 40
    mavg20 = sum(price8[-20:]) / 20

    trend = 1 if mavg20 > mavg40 else -1
    # print("Trend: ", trend, preTrend)
    
    # If the 2 extremes above doesn't work, then buy/sell everything
    # when there is a change in trend
    if (trend == 1 and preTrend in (-1, 0) and price8[-1] < priceMean):
        currentPos[8] = setVolume(INF, price8[-1])
    elif (trend == -1 and preTrend in (1, 0) and price8[-1] > priceMean):
        currentPos[8] = setVolume(-INF, price8[-1])
    # Update preTrend
    preTrend = trend
    

    # Revert everything in the last timestamp
    if (len(price8) == 250):
        print("Revert. Price: ", price8[-1])
        print("Price: ", price8[-1], "for", currentPos[8])
        print("Value:", price8[-1] * currentPos[8])
    

# Very simple mean reversion
# PL: 2, std: 14.97, score: 0.47
def meanRevert8(prices):
    global currentPos, limit

    print(currentPos[8])
    if (len(prices[8]) <= 4):
        return

    price8 = prices[8]
    priceStd = 0.585843
    priceMean = 68.537300

    diff = price8[-1] - priceMean
    upper = 1.5 * priceStd
    mid = upper/2

    limitVol = 2 * limit/price8[-1]
    
    # Sell when above mean
    if diff >= upper:
        currentPos[8] = setVolume(-limitVol, price8[-1])
        return
    elif diff >= mid:
        currentPos[8] = setVolume(currentPos[8] - limitVol * (diff - mid)/(upper - mid), price8[-1])
        return
    
    # Buy when below mean
    if diff <= -upper:
        currentPos[8] = setVolume(limitVol, price8[-1])
        return
    elif diff <= -mid:
        currentPos[8] = setVolume(currentPos[8] + limitVol * (-diff - mid)/(upper - mid), price8[-1])
        return



def getMyPosition(prices):
    global currentPos
    nins, nt = prices.shape
    if nt < 2:
        return np.zeros(nins)

    # Terry's part
    # Clean up Terry's param. TODO: Recheck this
    # better_pair(prices, 11, 42, 2.81, -9.48586161, -8.89586161, -8.30586161, 8, 21)
    # better_pair(prices, 14, 30, 0.7192064351085297, 4.86640231, 5.15640231, 5.44640231, 6, 4)
    # better_pair(prices, 22, 24, 3.4117759223360125, -149.87702011, -148.68702011, -147.49702011, 8, 27)
    # better_pair(prices, 28, 41, 0.021148, 49.6408878, 49.8208878, 50.0008878, 48, 1)
    # better_pair(prices, 7, 19, 0.3973201339866572, 33.5724356, 34.1524356, 34.7324356, 13, 5)
    # better_pair(prices, 43, 49, 0.2046650849773665, 47.71489042555398, 48.64489042555398, 49.57489042555398, 10, 2)
    # better_pair(prices, 13, 39, -2.935182925012027, 193.2674924878151, 193.84749248781512, 194.42749248781513, 7.0, 19.0)
    # better_pair(prices, 12, 34, 0.23829431834048995, 20.657766061851223, 20.757766061851225, 20.857766061851226, 9.0, 2.0)
    # better_pair(prices, 12, 25, -0.06366267571277003, 29.863092873692395, 29.983092873692396, 30.103092873692397, 16.0, 1.0)
    # better_pair(prices, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299, 39.807363604822996, 12.0, 2.0)
    
    meanRevert8(prices)

    return currentPos
