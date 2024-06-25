from collections import defaultdict

import numpy as np
import pandas as pd

from pair_trading.pair_trading_max_vol import better_pair, better_pair_aggregate
from single_series.mean_revert import meanRevertStrict, meanRevertGradual, movingAvg

from single_series.arima import linreg
import statsmodels.api as sm

from util.util import setVolume
from util.constants import LIMIT, INF, COMM_RATE

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

currentPos = np.zeros(50)
# model4 = sm.load("./arima_model/model4.pickle")
model6 = sm.load("./arima_model/model6.pickle")

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
    
    better_pair(prices, 1, 10, 0.824083971539593, 40.3584735402981, 41.2184735402981, 42.0784735402981, 8.0, 6.0)
    # New pairs added: Improve the performance across min, mean, max, but also widen the std (so takes caution, prob re-check?)
    better_pair(prices, 28, 39, 1.2159226859939878, -8.963364425168958, -8.783364425168958, -8.603364425168959, 7.0, 8.0)
    better_pair(prices, 2, 11, 0.22442595136426546, 41.24598604371131, 41.35598604371131, 41.46598604371131, 9.0, 2.0)
    better_pair(prices, 5, 38, 1.5318067990822284, -21.340981495923447, -21.19098149592345, -21.04098149592345, 6.0, 8.0)
    # Mean PL jumps by 10 (and so is PL std), hmmm
    better_pair(prices, 29, 30, 0.45879271464201205, 25.24018140455108, 25.39018140455108, 25.540181404551078, 9.0, 4.0)

    currentPos = better_pair_aggregate()
    
    # Single trade
    # For ticker 8, simple mean reversion (TODO: See if there is a better way)
    # Increase performance by .1 
    meanRevertGradual(currentPos, prices, 8, 68.537300, 0.585843)
    
    # # Ticker 0: No clue
    # # Ticker 3: Moving average: Mean = 0.2, Std = 17. 
    # # Currently hurts average performance on avg all start points, so comment it
    # # movingAvg(currentPos, prices, 3, 48.004780, 2.051494, longPeriod=30, shortPeriod=15)

    # Ticker 4: Negative score, but positive PL. Performs poorly for sudden jump.
    # Hurst avg start points performance :| Fuk
    # # linreg(currentPos, prices, 0.1, model4, 4)

    # Ticker 6: Less risky, higher PL, but with higher Std, so lower score
    # meanRevertGradual(currentPos, prices, 6, 18.177200, 0.299771)
    # Hurt performance by -0.2
    # movingAvg(currentPos, prices, 6, 18.177200, 0.299771, 40, 20)

    # Ticker 27: Slow, significant trend 
    # movingAvg(currentPos, prices, 27, 28.912860, 0.495184, 40, 20)

    # Ticker 18: Slow, significant trend
    # Somehow still hurts performance
    # movingAvg(currentPos, prices, 18, 13.722540, 0.697558, 40, 20)

    return currentPos


if __name__ == "__main__":
    from custom_eval.alleval import all_eval
    all_eval(getMyPosition, 100)