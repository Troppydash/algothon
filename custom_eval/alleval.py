#!/usr/bin/env python

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from pair_trading.pair_trading_max_vol import init_better_pair
from single_series.mean_revert import init_movingAvg

nInst = 0
nt = 0
commRate = 0.0010
dlrPosLimit = 10000


def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return (df.values).T


def calcPL(getPosition, start, prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(start, start + 251):
        prcHistSoFar = prcHist[:, :t]
        newPosOrig = getPosition(prcHistSoFar)
        curPrices = prcHistSoFar[:, -1]
        posLimits = np.array([int(x) for x in dlrPosLimit / curPrices])
        newPos = np.clip(newPosOrig, -posLimits, posLimits)
        deltaPos = newPos - curPos
        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = dvolume * commRate
        cash -= curPrices.dot(deltaPos) + comm
        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume
        # print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
        #       (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)

def all_eval(currentPos, getPosition, limit = 250):
    pricesFile = "./prices.txt"
    prcAll = loadPrices(pricesFile)
    print("Loaded %d instruments for %d days" % (nInst, nt))

    meanpls = []
    plstds = []
    scores = []

    start = [224, 215, 203, 236, 233, 226, 230, 237, 246, 220, 225, 229, 213, 209, 210, 214, 234, 206, 242, 249, 204, 207, 227, 208, 244, 221, 248, 219, 222, 250, 223, 232, 238, 235, 200, 202, 241, 247, 211, 243, 218, 216, 228, 245, 217, 231, 239, 212, 201, 240, 205]
    for i in range(0, min(limit, len(start))):
        # Reset the current position
        for j in range(50):
            currentPos[j] = 0
        # Reset the global variable (probably should use class)
        init_better_pair()
        init_movingAvg()
        
        currentStart = start[i]
        (meanpl, ret, plstd, sharpe, dvol) = calcPL(getPosition, currentStart, prcAll)
        score = meanpl - 0.1*plstd

        meanpls.append(meanpl)
        plstds.append(plstd)
        scores.append(score)

        # print("=====")
        # print("mean(PL): %.1lf" % meanpl)
        # print("return: %.5lf" % ret)
        # print("StdDev(PL): %.2lf" % plstd)
        # print("annSharpe(PL): %.2lf " % sharpe)
        # print("totDvolume: %.0lf " % dvol)
        # print("Score: %.2lf" % score)
    
    print("Summary: ")
    print("Mean: ")
    print(pd.Series(meanpls).describe())
    plt.figure()
    plt.plot(meanpls)
    plt.show()

    print("Std: ")
    print(pd.Series(plstds).describe())
    print("Score")
    print(pd.Series(scores).describe())