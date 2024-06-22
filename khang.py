from collections import defaultdict

import numpy as np
import pandas as pd

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

currentPos = np.zeros(50)
limit = 10000



def getUnitTradeVolume(prices, share1, share2, trade1, trade2):
    global limit

    price1 = prices[-1][share1]
    price2 = prices[-1][share2]

    # Get available volume
    vol1 = limit/ price1
    vol2 = limit/ price2

    unitTrade = min(vol1/trade1, vol2/trade2)
    return unitTrade


last = defaultdict(lambda: 0)
def better_pair(prices, share1, share2, beta, lower, middle, upper, trade1, trade2):
    global last, currentPos
    
    delta = prices[-1][share1] - beta * prices[-1][share2]
    unitTrade = getUnitTradeVolume(prices, share1, share2, trade1, trade2)
    volume1 = unitTrade * trade1
    volume2 = unitTrade * trade2

    if delta < lower and last[share1,share2] != -1:
        last[share1,share2] = -1
        currentPos[share1] = int(volume1)
        currentPos[share2] = -int(volume2)
    
    if last[share1,share2] < 0 and delta > middle or last[share1,share2] > 0 and delta < middle:
        last[share1, share2] = 0
        currentPos[share1] = 0
        currentPos[share2] = 0

    if delta > upper and last[share1,share2] != 1:
        last[share1,share2] = 1
        currentPos[share1] = -int(volume1)
        currentPos[share2] = int(volume2)

# Clean up Terry's param. TODO: Recheck this
def getMyPosition(prices):
    global currentPos
    nins, nt = prices.shape
    if nt < 2:
        return np.zeros(nins)

    better_pair(prices, 11, 42, 2.81, -9.48586161, -8.89586161, -8.30586161, 8, 21)
    better_pair(prices, 14, 30, 0.7192064351085297, 4.86640231, 5.15640231, 5.44640231, 6, 4)
    better_pair(prices, 22, 24, 3.4117759223360125, -149.87702011, -148.68702011, -147.49702011, 8, 27)
    better_pair(prices, 28, 41, 0.021148, 49.6408878, 49.8208878, 50.0008878, 48, 1)
    better_pair(prices, 7, 19, 0.3973201339866572, 33.5724356, 34.1524356, 34.7324356, 13, 5)
    better_pair(prices, 43, 49, 0.2046650849773665, 47.71489042555398, 48.64489042555398, 49.57489042555398, 10, 2)
    better_pair(prices, 13, 39, -2.935182925012027, 193.2674924878151, 193.84749248781512, 194.42749248781513, 7.0, 19.0)
    better_pair(prices, 12, 34, 0.23829431834048995, 20.657766061851223, 20.757766061851225, 20.857766061851226, 9.0, 2.0)
    better_pair(prices, 12, 25, -0.06366267571277003, 29.863092873692395, 29.983092873692396, 30.103092873692397, 16.0, 1.0)
    better_pair(prices, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299, 39.807363604822996, 12.0, 2.0)
    

    return currentPos
