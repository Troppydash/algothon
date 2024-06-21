import numpy as np
import pandas as pd

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

currentPos = np.zeros(50)


def predict(returns):
    return returns[-1] * 0.0472 + returns[-2] * 0.0996

def set_value(df, ticker, value):
    val = df[ticker].values[-1]
    vol = int(value / val)
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

def pair_trade(df, t1, t2, mean, sd, clamp_z, z, v):
    spread = df[t1].values[-1] - df[t2].values[-1]
    norm = (spread - mean) / sd
    if abs(norm) < clamp_z:
        return
    
    exp = clamp(norm / z, -1, 1) * v
    currentPos[t1] = set_value(df, t1, -exp)
    currentPos[t2] = set_value(df, t2, exp)

def pair_trade_log(df, t1, t2, mean, sd, clamp_z, z, v):
    spread = np.log(df[t1].values[-1]) - np.log(df[t2].values[-1])
    norm = (spread - mean) / sd
    if abs(norm) < clamp_z:
        return
    
    # exp = clamp(norm / z, -1, 1) * v
    # currentPos[t1] = set_value(df, t1, -exp)
    # currentPos[t2] = set_value(df, t2, exp)
    exp = clamp(norm / z, -1, 1) * 180
    v1, v2 = set_volume(df, t1, -exp), set_volume(df, t2, exp)
    vmin = min(abs(v1), abs(v2))
    currentPos[t1] = vmin * np.sign(v1)
    currentPos[t2] = vmin * np.sign(v2)

def getMyPosition(prices):
    global currentPos
    nins, nt = prices.shape
    if nt < 2:
        return np.zeros(nins)

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    pair_trade_log(df, 28, 49, -0.09244661852645539, 0.018986374603144413, 2, 2.5, 9000)
    # pair_trade(df, 1, 10, 34.04245999999999, 1.603967876361618, 0.6, 2.5, 9000)
    pair_trade_log(df, 7, 8, -0.38218823, 0.02883, 2, 2.5, 9000)
    pair_trade_log(df, 43, 47, 0.5692941382086553, 0.03309117994281669, 1, 2.5, 9000)
    pair_trade_log(df, 22, 24, 0.06450, 0.032329, 1, 2.5, 9000)
    pair_trade_log(df, 14, 23, -0.697, 0.040451, 1, 2.5, 9000)
    pair_trade_log(df, 11, 42, 0.7556795088657547, 0.038963533345916526, 1, 2.5, 9000)
  

    # currentPos = np.array([int(x) for x in currentPos])
    return currentPos
