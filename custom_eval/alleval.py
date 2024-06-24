#!/usr/bin/env python

import numpy as np
import pandas as pd

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

def all_eval(getPosition):
    pricesFile = "./prices.txt"
    prcAll = loadPrices(pricesFile)
    print("Loaded %d instruments for %d days" % (nInst, nt))

    meanpls = []
    plstds = []
    scores = []

    for i in range(1, 251):
        # print("Start at: ", i)
        (meanpl, ret, plstd, sharpe, dvol) = calcPL(getPosition, i, prcAll)
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
    print("Std: ")
    print(pd.Series(plstds).describe())
    print("Score")
    print(pd.Series(scores).describe())