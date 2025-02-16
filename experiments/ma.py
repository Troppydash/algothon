from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.tsa.ardl as ardl
# from filterpy.kalman import KalmanFilter, predict, update
# from pykalman import KalmanFilter
from scipy.optimize import least_squares
from statsmodels.tsa.stattools import coint


def find_cointegrated_pairs(data):
    n = 50
    pairs = []
    found = set()
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[i]
            S2 = data[j]
            result = coint(S1, S2)
            pvalue = result[1]
            if pvalue < 0.005:
                if i in found or j in found:
                    continue

                found.add(i)
                found.add(j)
                pairs.append((i,j))
    return pairs

# fil = KalmanFilter(dim_x=2,dim_z=2)
# fil.x = [0, 1]
# fil.F = np.eye(2,2)

# fil = KalmanFilter(
#     transition_matrices=[[1,0],[0,1]],
#     observation_matrices=
# )

def find_param(data, t1, t2):
    pass


state = [0, 1]
P = np.zeros((2, 2))
F = np.eye(2)


def test_threshold(spread):
    n = 50
    s0 = np.linspace(0, np.max(spread), n)
    f_bar = np.array([None] * n)
    for i in range(n):
        f_bar[i] = (len(spread[spread > s0[i]]) + len(spread[spread < -s0[i]])) / spread.shape[0]

    D = np.zeros((n - 1, n))
    for i in range(D.shape[0]):
        D[i, i] = 1
        D[i, i + 1] = -1

    l = 1.0
    f_star = np.linalg.inv(np.eye(n) + l * D.T @ D) @ f_bar.reshape(-1, 1)
    s_star = [f_star[i] * s0[i] for i in range(n)]
    threshold = s0[s_star.index(max(s_star))]

    return threshold


def compute_volume(data, betas, tickers):
    unit = min(10000 / data[t].values[-1] / abs(b) for t, b in zip(tickers, betas))
    return [b * unit for b in betas]


states = defaultdict(lambda: [])
deltas = defaultdict(lambda: [])


def precompute_states(df, t1, t2, i):
    data = df[[t1, t2]].iloc[:i].iloc[-300:]

    # find mu and gamma
    n = len(data[t1].values)
    result = least_squares(
        lambda x: np.log(data[t1].values) - (x[0] * np.ones(n) + x[1] * np.log(data[t2].values)),
        x0=[0, 1]
    )
    states[t1, t2].append(result.x)

    if len(states[t1, t2]) < 5:
        mean, gamma = result.x
    else:
        ok = pd.DataFrame(np.array(states[t1, t2]), columns=[0, 1])
        mean, gamma = ok.ewm(span=5).mean().iloc[-1]
        # mean, gamma = np.array(states[t1, t2])[-10:].sum(axis=0) / 10

    y1, y2 = np.log(data[t1].values), np.log(data[t2].values)
    deltas[t1, t2].append(y1[-1] - mean - gamma * y2[-1])


def kalman_trade(df, t1, t2):
    data = df[[t1, t2]].iloc[-300:]

    # find mu and gamma
    n = len(data[t1].values)
    result = least_squares(
        lambda x: np.log(data[t1].values) - (x[0] * np.ones(n) + x[1] * np.log(data[t2].values)),
        x0=[0, 1]
    )
    states[t1, t2].append(result.x)

    ok = pd.DataFrame(np.array(states[t1, t2]), columns=[0,1])
    mean, gamma = ok.ewm(span=5).mean().iloc[-1]
    # mean, gamma = np.array(states[t1, t2])[-10:].sum(axis=0) / 10
    y1, y2 = np.log(data[t1].values), np.log(data[t2].values)


    threshold = test_threshold(np.array(deltas[t1, t2])[-100:])
    beta = np.array([1, -gamma])
    deltas[t1, t2].append(y1[-1] - mean - gamma * y2[-1])

    beta /= np.sum(np.abs(beta))

    vol = compute_volume(data, beta, [t1, t2])
    if deltas[t1, t2][-1] > threshold:
        # sell
        currentPos[t1] = -int(vol[0])
        currentPos[t2] = -int(vol[1])

    elif deltas[t1, t2][-1] < -threshold:
        # buy
        currentPos[t1] = int(vol[0])
        currentPos[t2] = int(vol[1])
    #
    # elif currentPos[t1] * -int(vol[0]) > 0 and deltas[t1, t2][-1] < 0 \
    #         or currentPos[t1] * int(vol[0]) > 0 and deltas[t1, t2][-1] > 0:
    #     currentPos[t1] = currentPos[t2] = 0


currentPos = np.zeros(50)

index = 0

pairs = [(1, 10), (7, 24), (27, 36), (28, 39)]


def getMyPosition(prices):
    global currentPos, oldPos, index, pairs

    nins, nt = prices.shape

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    oldPos = np.copy(currentPos)


    # if nt % 69 == 0 or index == 0:
    #     pairs = find_cointegrated_pairs(df.apply(np.log).iloc[-300:])
    #     print(pairs)
    #     currentPos = np.zeros(50)
    #     for i in range(nt-300, nt):
    #         for pair in pairs:
    #             precompute_states(df, pair[0], pair[1], i)
    # index += 1
    pairs = [(11,42),(2,36),(1,10),(4,32),(24,49),(22,47)]
    if index == 0:
        for i in range(nt-200, nt):
            for pair in pairs:
                precompute_states(df, pair[0], pair[1], i)
    index += 1

    if True:
        # try 11 42
        # for pair in pairs:
        #     kalman_trade(df, pair[0], pair[1])
        kalman_trade(df, 11, 42)
        kalman_trade(df, 2, 36)
        kalman_trade(df, 1,10)
        kalman_trade(df,  4, 32)
        kalman_trade(df,  24,49)
        kalman_trade(df,  22, 47)

    return currentPos
#
#
# if __name__ == "__main__":
#     from custom_eval.alleval import all_eval
#
#     all_eval(currentPos, getMyPosition, 10)
