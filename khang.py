from collections import defaultdict

import numpy as np
import pandas as pd

from pair_trading.pair_trading_max_vol import better_pair, better_pair_aggregate
from single_series.mean_revert import meanRevert

from util.util import setVolume
from util.constants import LIMIT, INF, COMM_RATE

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

currentPos = np.zeros(50)


def getMyPosition(prices):
    global currentPos
    nins, nt = prices.shape
    if nt < 2:
        return np.zeros(nins)

    # Terry's part
    # Clean up Terry's param. TODO: Recheck this
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
    # better_pair(prices, 1, 10, 0.824083971539593, 40.3584735402981, 41.2184735402981, 42.0784735402981, 8.0, 6.0)

    currentPos = better_pair_aggregate()
    
    # Single trade
    # For ticker 8
    meanRevert(currentPos, prices, 8, 68.537300, 0.585843)

    return currentPos


if __name__ == "__main__":
    from custom_eval.alleval import all_eval
    all_eval(getMyPosition)