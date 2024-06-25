#!/usr/bin/env python

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

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

def all_eval(getPosition, limit = 250):
    pricesFile = "./prices.txt"
    prcAll = loadPrices(pricesFile)
    print("Loaded %d instruments for %d days" % (nInst, nt))

    meanpls = []
    plstds = []
    scores = []

    start = [198, 215, 180, 9, 209, 111, 194, 122, 144, 34, 104, 96, 3, 133, 125, 86, 40, 79, 71, 41, 57, 128, 241, 108, 182, 69, 95, 124, 232, 89, 250, 192, 67, 58, 55, 7, 101, 19, 109, 4, 185, 92, 211, 47, 228, 76, 87, 166, 1, 242, 222, 129, 13, 176, 234, 165, 139, 134, 117, 157, 205, 236, 48, 195, 14, 42, 154, 153, 68, 179, 35, 37, 231, 223, 66, 78, 148, 10, 119, 25, 80, 43, 132, 208, 18, 197, 53, 169, 201, 72, 45, 103, 54, 94, 64, 200, 173, 83, 207, 167, 123, 247, 186, 235, 187, 164, 214, 38, 102, 36, 17, 249, 171, 202, 32, 174, 221, 212, 52, 151, 22, 145, 149, 184, 244, 191, 39, 99, 27, 135, 59, 142, 137, 140, 118, 6, 152, 127, 193, 20, 159, 220, 8, 141, 206, 227, 81, 218, 114, 130, 131, 56, 77, 65, 107, 85, 246, 163, 158, 146, 26, 143, 168, 105, 50, 91, 29, 217, 177, 138, 213, 5, 170, 112, 199, 61, 126, 63, 204, 110, 88, 238, 49, 224, 46, 155, 219, 121, 239, 150, 245, 31, 189, 203, 62, 181, 161, 188, 160, 84, 196, 75, 183, 230, 190, 229, 225, 136, 74, 60, 15, 156, 51, 178, 12, 33, 172, 243, 106, 70, 216, 248, 2, 97, 147, 23, 44, 115, 113, 175, 233, 21, 210, 90, 226, 100, 162, 240, 24, 116, 11, 93, 120, 28, 73, 30, 237, 82, 16, 98]
    for i in range(0, limit):
        currentStart = start[i]
        # print("Start at: ", i)
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