from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.tsa.ardl as ardl
# from filterpy.kalman import KalmanFilter, predict, update
from pykalman import KalmanFilter
from scipy.optimize import least_squares
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


# fil = KalmanFilter(dim_x=2,dim_z=2)
# fil.x = [0, 1]
# fil.F = np.eye(2,2)

# fil = KalmanFilter(
#     transition_matrices=[[1,0],[0,1]],
#     observation_matrices=
# )

def find_param(data, t1, t2):
    pass


# state = [0, 1]
# P = np.zeros((2, 2))
# F = np.eye(2)


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


def KalmanFilterRegression(df, t1, t2):
    obs_mat = sm.add_constant(df[t1].values, prepend=False)[:, np.newaxis]

    kf = KalmanFilter(
        n_dim_obs=1, n_dim_state=2,
        initial_state_mean=np.ones(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=0.2,
        transition_covariance=0.01 * np.eye(2)
        # em_vars=['observation_covariance', 'transition_covariance']
    )
    state_means, state_covs = kf.filter(df[t2].values)
    return state_means


def kalman_trade(df, t1, t2):
    data = df[[t1, t2]]

    result = KalmanFilterRegression(data, t1, t2)
    gamma, mean = result[-1, :]
    delta = data[t2] - (data[t1] * gamma + mean)
    # zscore = delta

    # threshold = test_threshold(np.array(delta.values)[-500:])

    zscore = (delta - np.mean(delta)) / np.std(delta)
    deltas[t1, t2].append(zscore.values[-1])
    beta = np.array([1, -gamma])
    beta /= np.sum(np.abs(beta))

    vol = compute_volume(data, beta, [t1, t2])
    entry = 1.9

    if zscore.values[-1] < -entry:
        currentPos[t1] = int(vol[0])
        currentPos[t2] = int(vol[1])

    # if zscore.values[-1] < 0 and zscore.values[-2] > 0:
    #     currentPos[t1] = 0
    #     currentPos[t2] = 0

    if zscore.values[-1] > entry:
        currentPos[t1] = -int(vol[0])
        currentPos[t2] = -int(vol[1])

    # if zscore.values[-1] > 0 and zscore.values[-2] < 0:
    #     currentPos[t1] = 0
    #     currentPos[t2] = 0

    # if deltas[t1, t2][-1] > threshold:
    #     # sell
    #     currentPos[t1] = -int(vol[0])
    #     currentPos[t2] = -int(vol[1])
    #
    # elif deltas[t1, t2][-1] < -threshold:
    #     # buy
    #     currentPos[t1] = int(vol[0])
    #     currentPos[t2] = int(vol[1])
    #
    # elif currentPos[t1] * -int(vol[0]) > 0 and deltas[t1, t2][-1] < 0 \
    #         or currentPos[t1] * int(vol[0]) > 0 and deltas[t1, t2][-1] > 0:
    #     currentPos[t1] = currentPos[t2] = 0


currentPos = np.zeros(50)

index = 0


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
            if pvalue < 0.01:
                if i in found or j in found:
                    continue

                found.add(i)
                found.add(j)
                pairs.append((i,j))
    return pairs

pairs = []

def getMyPosition(prices):
    global currentPos, oldPos, index, pairs

    nins, nt = prices.shape

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    oldPos = np.copy(currentPos)

    # if nt % 50 == 0:
    #     pairs = find_cointegrated_pairs(df.iloc[-250:])
    #     currentPos = np.zeros(50)
    # pairs = [(7, 24)]
    # pairs = [(1, 10)]
    # pairs = [(1,10),(7,24),(20,34),(28,39),(27,36),(31,40),(43,46)]

    # if index == 0:
    #     for i in range(400, nt):
    #         for pair in pairs:
    #             precompute_states(df, pair[0], pair[1], i)
    # index += 1

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
