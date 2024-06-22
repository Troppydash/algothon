from collections import defaultdict

import numpy as np
import pandas as pd

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

currentPos = np.zeros(50)


def set_value(df, ticker, value):
    value = clamp(value, -10000, 10000)
    val = df[ticker].values[-1]
    vol = int(value / val)

    assert -10000 <= vol * val <= 10000
    return vol


def set_volume(df, ticker, value):
    val = df[ticker].values[-1]
    if abs(val * value) > 10000:
        return int(10000 / val)

    return value


def clamp(value, lower, upper):
    if value < lower:
        return lower

    if value > upper:
        return upper

    return value


#
#
# OUT = 'out'
# IN = 'in'

#
# class MeanReversion:
#
#     def __init__(self, spread=80.0, c=0.05, mean=5000, limit=20):
#         self.mean = mean
#         self.spread = spread
#         self.limit = limit
#         self.c = c
#         self.state = OUT
#
#     def outwards(self, x):
#         if x == 0:
#             return 0
#
#         if x > 0:
#             return -(x ** self.c)
#
#         return (-x) ** self.c
#
#     def inwards(self, x):
#         if x == 0:
#             return 0
#
#         if x > 0:
#             return -(x ** (1 / self.c))
#
#         return (-x) ** (1 / self.c)
#
#     def create_spread(self, new_price, old_price, pos) -> int:
#         mean = self.mean
#
#         outw = int(self.limit * self.outwards((new_price - mean) / self.spread))
#         inw = int(self.limit * self.inwards((new_price - mean) / self.spread))
#
#         if old_price - mean > 0 and new_price - mean < 0:
#             self.state = OUT
#         if old_price - mean < 0 and new_price - mean > 0:
#             self.state = OUT
#
#         if new_price - mean > 0:
#             # we are positive delta
#
#             if new_price > old_price:
#                 # we more positive
#
#                 if self.state == OUT:
#                     return outw
#                 else:
#                     # check if we've reached it
#                     if outw < pos:
#                         self.state = OUT
#                         return outw
#                     else:
#                         return pos
#             else:
#                 # we more negative
#                 if self.state == IN:
#                     return inw
#                 else:
#                     # check if we've reached it
#                     if inw > pos:
#                         self.state = IN
#                         return inw
#                     else:
#                         return pos
#
#         else:
#             # we are negative delta
#
#             if new_price < old_price:
#                 # if more negative
#
#                 if self.state == OUT:
#                     return outw
#                 else:
#                     if outw > pos:
#                         self.state = OUT
#                         return outw
#                     else:
#                         return pos
#             else:
#                 # if more positive
#
#                 if self.state == IN:
#                     return inw
#                 else:
#                     if inw < pos:
#                         self.state = IN
#                         return inw
#                     return pos
#
#
# def pair_trade_log(df, t1, t2, mean, sd, clamp_z, z, v):
#     spread = np.log(df[t1].values[-1]) - np.log(df[t2].values[-1])
#     norm = (spread - mean) / sd
#     if abs(norm) < clamp_z:
#         return
#
#     # exp = clamp(norm / z, -1, 1) * v
#     # currentPos[t1] = set_value(df, t1, -exp)
#     # currentPos[t2] = set_value(df, t2, exp)
#     exp = clamp(norm / z, -1, 1) * 170
#     v1, v2 = set_volume(df, t1, -exp), set_volume(df, t2, exp)
#     vmin = min(abs(v1), abs(v2))
#     currentPos[t1] = vmin * np.sign(v1)
#     currentPos[t2] = vmin * np.sign(v2)


# m = 0.75567
# sd = 0.03896
# mr = MeanReversion(spread=2.2*sd, mean=m, limit=600, c=3)

# m = 0.03865
# sd = 0.005758
# mr2 = MeanReversion(spread=1.5*sd, mean=m, limit=300, c=3)
#
# def another_pair(df, mr, t1, t2):
#     delta = np.log(df[t1].values[-1]) - np.log(df[t2].values[-1])
#     prev = np.log(df[t1].values[-2]) - np.log(df[t2].values[-2])
#     p = currentPos[t1] - currentPos[t2]
#     pos = mr.create_spread(delta, prev, p)
#     currentPos[t1] = pos // 2
#     currentPos[t2] = -(pos // 2)


last = defaultdict(lambda: 0)


def better_pair(df, t1, t2, beta, lower, middle, upper, s1, s2, v1, v2, scaler: float = 1.1):
    global last

    delta = df[t1].values[-1] - beta * df[t2].values[-1]

    if delta < lower and last[t1, t2] != -1:
        last[t1, t2] = -1
        currentPos[t1] = int(s1 * v1 * scaler)
        currentPos[t2] = -int(s2 * v2 * scaler)
    #
    if last[t1, t2] < 0 and delta >= middle or last[t1, t2] > 0 and delta <= middle:
        last[t1, t2] = 0
        currentPos[t1] = currentPos[t1] // 2
        currentPos[t2] = currentPos[t2] // 2

    if delta > upper and last[t1, t2] != 1:
        last[t1, t2] = 1
        currentPos[t1] = -int(s1 * v1 * scaler)
        currentPos[t2] = int(s2 * v2 * scaler)


def regularize(df):
    for i in range(50):
        pos, val = currentPos[i], df[i].values[-1]
        if pos * val > 10000:
            currentPos[i] = int(10000 / val)
        if pos * val < -10000:
            currentPos[i] = int(-10000 / val)


#
# def best_pair(df, weights, ts, lower, middle, upper):
#     global net
#
#     delta = 0
#     for w, t in zip(weights, ts):
#         delta += w * df[t].values[-1]
#
#     if delta < lower and net != -1:
#         net = -1
#         for w, t in zip(weights, ts):
#             currentPos[t] = int(100 * w)
#
#     if net == -1 and delta > middle or net == 1 and delta < middle:
#         net = 0
#         for w, t in zip(weights,ts):
#             currentPos[t] = 0
#
#     if delta > upper and net != 1:
#         net = 1
#         for w, t in zip(weights, ts):
#             currentPos[t] = int(-100 * w)


def getMyPosition(prices):
    global currentPos
    nins, nt = prices.shape
    if nt < 2:
        return np.zeros(nins)

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    weights = [-0.1344368, -0.035686, -0.1482424, -0.1665422, -0.0748774, 0.9619431]
    indices = [5, 13, 16, 28, 30, 38]
    # best_pair(df, weights, indices, 2.586489376879487, 2.6713907573920013, 2.7562921379045155)

    used = [11, 42, 14, 30, 22, 24, 28, 41, 7, 19, 43, 49, 13, 39, 12, 34, 12, 25, 7, 48]
    # better_pair(df, 28, 39, 1.2159226859939878, -8.963364425168958, -8.783364425168958, -8.603364425168959,
    #             7.0, 8.0, 27.414535186555913, 25.145845906256287)
    better_pair(df, 11, 42, 2.81, -9.48586161, -8.89586161, -8.30586161, 8, 21, 37, 32)
    better_pair(df, 14, 30, 0.7192064351085297, 4.86640231, 5.15640231, 5.44640231, 6, 4, 98, 162)
    better_pair(df, 22, 24, 3.4117759223360125, -149.87702011, -148.68702011, -147.49702011, 8, 27, 17, 5)
    better_pair(df, 28, 41, 0.021148, 49.6408878, 49.8208878, 50.0008878, 48, 1, 3, 129)
    better_pair(df, 7, 19, 0.3973201339866572, 33.5724356, 34.1524356, 34.7324356, 13, 5, 15, 56)
    better_pair(df, 43, 49, 0.2046650849773665, 47.71489042555398, 48.64489042555398, 49.57489042555398, 10.0,
                2.0, 15.130882130428203, 84.30281571404484)
    better_pair(df, 13, 39, -2.935182925012027, 193.2674924878151, 193.84749248781512, 194.42749248781513,
                7.0, 19.0, 27.78240817914097, 10.587724592107909)
    better_pair(df, 12, 34, 0.23829431834048995, 20.657766061851223, 20.757766061851225, 20.857766061851226,
                9.0, 2.0, 40.24306813151435, 175.13134851138352)
    better_pair(df, 12, 25, -0.06366267571277003, 29.863092873692395, 29.983092873692396, 30.103092873692397,
                16.0, 1.0, 22.63672582397682, 140.66676044450696)
    better_pair(df, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299, 39.807363604822996, 12.0,
                2.0, 16.311085013375088, 99.96001599360255)

    # 20,25 done
    # 21,34
    # 21,35
    # 24 41
    # 25 34
    # 27 47
    # 19 37
    # 18 36
    # 17 18
    # 16 25
    # 13 39
    # 12 34
    # 12 25
    # 12 15
    # 11 47
    # 11 20
    # 8 14
    # 7 48
    # 7 45
    # 7 32
    # 5 39
    # 4 47
    # 4 27
    # 4 11

    # better_pair(df, 5, 44, 0.15299458297216598, 6.19603119, 6.35603119, 6.51603119, 7, 1, 109, 255)
    # better_pair(df, currentPos[7]-currentPos[39], 7, 39, -1.1096567908341244, 100.64737749, 101.39737749, 102.14737749)
    # pair_trade_log(df, 28, 49, -0.09244661852645539, 0.018986374603144413, 2, 2.5, 9000)
    # pair_trade(df, 1, 10, 34.04245999999999, 1.603967876361618, 0.6, 2.5, 9000)
    # pair_trade_log(df, 7, 8, -0.38218823, 0.02883, 2, 2.5, 9000)
    # pair_trade_log(df, 43, 47, 0.5692941382086553, 0.03309117994281669, 1, 2.5, 9000)
    # pair_trade_log(df, 22, 24, 0.06450, 0.032329, 1, 2.5, 9000)
    # pair_trade_log(df, 14, 23, -0.697, 0.040451, 1, 2.5, 9000)
    # pair_trade_log(df, 28, 39, 0.03865, 0.005758, 1, 1.3, 9000)
    # another_pair(df, mr, 11, 42)
    # another_pair(df, mr2, 28, 39)
    # print((delta-m)/sd, pos)
    # z = (delta-m)/sd
    # if z > 0:
    #     print('11 > 42, short 11, buy 42')
    #     print(z, pos, currentPos[11], currentPos[42], mr.state)
    # else:
    #     print('11 < 42, buy 11, short 42')
    #     print(z, pos, currentPos[11], currentPos[42], mr.state)
    # pair_trade_log(df, 11, 42, 0.7556795088657547, 0.038963533345916526, 1, 2.5, 9000)

    # currentPos = np.array([int(x) for x in currentPos])
    regularize(df)

    return currentPos
