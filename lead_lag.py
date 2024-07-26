# import numpy as np
# import pandas as pd

# import statsmodels.tsa.ardl as ardl

# # UTIL
# # CONSTANTS
# LIMIT = 10000
# INF = 1000000000
# COMM_RATE = 0.0010

# currentPos = np.zeros(50)

# # LEAD-LAG STRATEGY
# def predict(currentPos, df, ticker, indices, lags, deg=1):

#     result = ardl.ARDL(
#         df[ticker].pct_change().dropna().values[-600:],
#         0,
#         df[indices].pct_change().dropna().values[-600:],
#         lags,
#         causal=True
#     ).fit()

#     predict_mu = result.forecast()
#     est = predict_mu
#     val = df[ticker].values[-1]
#     vol = int(10000 / val)
#     trans = 0.001 * deg * val * vol

#     pforecast = est * val
#     if abs(vol * pforecast) > trans:
#         if pforecast > 0:
#             currentPos[ticker] = vol
#         else:
#             currentPos[ticker] = -vol


# midPairs = [('25', '33', 573),
#  ('7', '15', 573),
#  ('31', '10', 286),
#  ('14', '43', 287),
#  ('43', '22', 286),
#  ('37', '45', 573),
#  ('9', '17', 573),
#  ('17', '25', 573),
#  ('42', '0', 572),
#  ('45', '3', 572),
#  ('30', '9', 286),
#  ('49', '7', 572),
#  ('41', '20', 286),
#  ('20', '49', 287),
#  ('36', '15', 286),
#  ('3', '11', 573),
#  ('11', '19', 573),
#  ('13', '42', 287),
#  ('15', '44', 287),
#  ('46', '25', 286),
#  ('35', '14', 286),
#  ('22', '30', 573),
#  ('19', '48', 287),
#  ('23', '31', 573),
#  ('33', '41', 573),
#  ('6', '14', 573),
#  ('1', '9', 573),
#  ('40', '19', 286),
#  ('34', '42', 573),
#  ('44', '23', 286)]

# safePairs = [('31', '10', 286),
#  ('14', '43', 287),
#  ('43', '22', 286),
#  ('30', '9', 286),
#  ('41', '20', 286),
#  ('20', '49', 287),
#  ('36', '15', 286),
#  ('7', '36', 287),
#  ('13', '42', 287),
#  ('15', '44', 287),
#  ('46', '25', 286),
#  ('35', '14', 286)]



# def getMyPosition(prices):
#     global currentPos

#     df = pd.DataFrame(prices.T, columns=np.arange(50))

#     # for i in range(42):
#     #     if i in [19,27,38]:
#     #         continue

#     #     predict(currentPos, df, i, [i+8], np.arange(572,575), 1)

#     # for i in range(42,50):
#     #     predict(currentPos, df, i, [i-42], np.arange(571,574), 1)

#     # predict(currentPos, df, 19, [48], np.arange(286, 289), 1)

#     # # Filtered pair
#     # for pair in safePairs:
#     #     predict(currentPos, df, int(pair[0]), [int(pair[1])], [pair[2]-1, pair[2], pair[2]+1])
 

#     return currentPos

import numpy as np
import pandas as pd

import statsmodels.tsa.ardl as ardl

# UTIL
# CONSTANTS
LIMIT = 10000
INF = 1000000000
COMM_RATE = 0.0010

currentPos = np.zeros(50)

memo = {}


# LEAD-LAG STRATEGY
def predict(currentPos, df, pct, ticker, indices, lags, deg):
    if ticker not in memo:
        result = ardl.ARDL(
            df[ticker].pct_change().dropna().values[:],
            0,
            df[indices].pct_change().dropna().values[:],
            lags,
            causal=True
        ).fit()

        memo[ticker] = result.params

    param = memo[ticker]

    predict_mu = np.sum(pct[indices].iloc[lags * -1, :].values.T.flatten() * param[1:]) + param[0]
    val = df[ticker].values[-1]
    est = predict_mu
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    pforecast = est * val
    if abs(vol * pforecast) > trans:
        if pforecast > 0:
            currentPos[ticker] = vol
        else:
            currentPos[ticker] = -vol


def getMyPosition(prices):
    global currentPos

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    pcf = df.pct_change().dropna()
    for i in range(50):
        if i in [2, 6, 16, 27, 38, 39, 48]:
            continue

        predict(currentPos, df, pcf, i, list(range(50)), np.array([1, 286, 287, 288]), 1)

    return currentPos
