from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from pair_trading.pair_trading_max_vol import better_pair, better_pair_aggregate

from single_series.mean_revert import meanRevertStrict, meanRevertGradual, movingAvg
from single_series.arima import linreg

from util.util import setVolume
from util.util import safetyCheck
from util.constants import LIMIT, INF, COMM_RATE

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

priceMeans = {
    0 : 13.845619999999998,
    1 : 69.03374000000001,
    2 : 47.26258,
    3 : 48.00478,
    4 : 55.496120000000005,
    5 : 11.7407,
    6 : 18.177199999999996,
    7 : 46.7828,
    8 : 68.5373,
    9 : 50.185100000000006,
    10 : 34.991279999999996,
    11 : 26.655479999999997,
    12 : 26.612,
    13 : 48.495039999999996,
    14 : 14.77168,
    15 : 25.024019999999997,
    16 : 35.25014,
    17 : 45.08558,
    18 : 13.722539999999999,
    19 : 30.998019999999997,
    20 : 64.69341999999999,
    21 : 22.44104,
    22 : 67.5549,
    23 : 29.657559999999997,
    24 : 63.297619999999995,
    25 : 54.63196,
    26 : 59.677859999999995,
    27 : 28.912860000000002,
    28 : 51.157979999999995,
    29 : 31.24322,
    30 : 13.47922,
    31 : 66.98914,
    32 : 54.52396000000001,
    33 : 42.46486,
    34 : 24.871520000000004,
    35 : 67.15796,
    36 : 36.1705,
    37 : 38.82432,
    38 : 21.555519999999998,
    39 : 49.21754,
    40 : 31.651400000000002,
    41 : 63.25164000000001,
    42 : 12.502360000000001,
    43 : 60.843540000000004,
    44 : 36.52358,
    45 : 52.46146,
    46 : 57.53716,
    47 : 34.42736,
    48 : 41.27304,
    49 : 56.123619999999995
}

priceStds = {
    0 : 0.7487449857644556,
    1 : 3.9081870047027505,
    2 : 0.6144378009387482,
    3 : 2.0514944334364746,
    4 : 1.7339159942021123,
    5 : 0.6358842721873459,
    6 : 0.2997710549248235,
    7 : 1.266560108726797,
    8 : 0.5858430969709701,
    9 : 3.207362176971684,
    10 : 3.131638593797874,
    11 : 2.4741732834369032,
    12 : 0.5680646044153597,
    13 : 1.9269707394928768,
    14 : 0.8162026409551763,
    15 : 1.1686007423367402,
    16 : 0.4320536089909262,
    17 : 0.46156640984955,
    18 : 0.6975581144579226,
    19 : 2.3644529081183703,
    20 : 5.786327996836503,
    21 : 0.5517758889833141,
    22 : 2.474040570299318,
    23 : 1.5613213298325173,
    24 : 0.8466383694691109,
    25 : 9.948234138949823,
    26 : 4.129548445184079,
    27 : 0.4951840414126242,
    28 : 0.38532930975194174,
    29 : 0.7848683126080914,
    30 : 0.76818031319943,
    31 : 2.2172782653993672,
    32 : 2.1043823722739825,
    33 : 2.651297897536783,
    34 : 1.8630766486641377,
    35 : 10.17399521307098,
    36 : 2.9614391423068573,
    37 : 1.8938973128613195,
    38 : 0.2559236672269562,
    39 : 0.231601872303548,
    40 : 0.771222671358691,
    41 : 7.833634211547067,
    42 : 0.9402253335001727,
    43 : 1.898022087355196,
    44 : 1.5619817404550635,
    45 : 0.7562138501603557,
    46 : 4.548283819839261,
    47 : 0.8781686573668883,
    48 : 3.8149081518188126,
    49 : 1.1780585467110527
}



currentPos = np.zeros(50)
# model4 = sm.load("./arima_model/model4.pickle")
# model6 = sm.load("./arima_model/model6.pickle")

def getMyPosition(prices):
    global currentPos
    nins, nt = prices.shape
    if nt < 2:
        return np.zeros(nins)

    # # Terry's part
    # # Clean up Terry's param. TODO: Recheck this
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
    
    # better_pair(prices, 1, 10, 0.824083971539593, 40.3584735402981, 41.2184735402981, 42.0784735402981, 8.0, 6.0)
    # # New pairs added: Improve the performance across min, mean, max, but also widen the std (so takes caution, prob re-check?)
    # better_pair(prices, 28, 39, 1.2159226859939878, -8.963364425168958, -8.783364425168958, -8.603364425168959, 7.0, 8.0)
    # better_pair(prices, 2, 11, 0.22442595136426546, 41.24598604371131, 41.35598604371131, 41.46598604371131, 9.0, 2.0)
    # better_pair(prices, 5, 38, 1.5318067990822284, -21.340981495923447, -21.19098149592345, -21.04098149592345, 6.0, 8.0)
    # # Mean PL jumps by 10 (and so is PL std), hmmm
    # better_pair(prices, 29, 30, 0.45879271464201205, 25.24018140455108, 25.39018140455108, 25.540181404551078, 9.0, 4.0)

    # currentPos = better_pair_aggregate(currentPos)
    
    # # Single trade
    # # For ticker 8, simple mean reversion (TODO: See if there is a better way)
    # # Increase performance by .1 
    # meanRevertGradual(currentPos, prices, 8, 68.537300, 0.585843)

    # Ticker 27: Slow, significant trend. Increase performance by .02 
    # movingAvg(currentPos, prices, 27, 28.912860, 0.495184, 40, 20, 0.1)

    # Testing: Risky, not-working stuff
    
    # # Ticker 0: No clue

    # Ticker 3: Moving average
    # movingAvg(currentPos, prices, 3, 48.004780, 2.051494, longPeriod=40, shortPeriod=20, threshold=0.2)

    # # Ticker 4: Negative score, but positive PL. Performs poorly for sudden jump.
    # # Hurst avg start points performance :| Fuk
    # linreg(currentPos, prices, 0.1, model4, 4)

    # # Ticker 6: Less risky, higher PL, but with higher Std, so lower score
    # meanRevertGradual(currentPos, prices, 6, 18.177200, 0.299771)
    # # Hurt performance by -0.2
    # movingAvg(currentPos, prices, 6, 18.177200, 0.299771, 40, 20)

    # # Ticker 18: Significant trend, but initial jerky part. 
    # # Can't predict trend reliably (due to the initial jerking part)
    # movingAvg(currentPos, prices, 18, 13.722540, 0.697558, 30, 15, threshold=0.1)

    # Run safety check on all tickers
    for i in range(50):
        safetyCheck(currentPos, prices, i, priceMeans[i], priceStds[i])

    return currentPos


if __name__ == "__main__":
    from custom_eval.alleval import all_eval
    all_eval(currentPos, getMyPosition, 250)