#!/usr/bin/env python

import numpy as np
import pandas as pd
from team import getMyPosition as getPosition
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


pricesFile = "./prices.txt"
prcAll = loadPrices(pricesFile)
print("Loaded %d instruments for %d days" % (nInst, nt))

start = np.random.randint(1, 250)
start = 250
ticker = 32

values = []
prices = []
volumes = []

# pairs = [(28,39), (11,42),(43,49),(15,20),(1,10),(7,8),(14,30),(22,24),(25,36),(31,40),(33,37),(4,32),(9,46),(13,45),(44,47)]
# pairs = [(12,34),(12,25),(7,48), (7,19), (28,39), (28,41), (2,11), (11,42), (43,49),(14,30),(22,24)]
pairs = [(12,34),(7,48), (28,39), (2,11), (43,49),(14,30),(22,24)]
pnls = {f"{t1}-{t2}": [0] for t1,t2 in pairs}
positions = {f"{t1}-{t2}": [(0,0)] for t1,t2 in pairs}

def calcPL(prcHist):
    cash = 0
    curPos = np.zeros(nInst)
    totDVolume = 0
    totDVolumeSignal = 0
    totDVolumeRandom = 0
    value = 0
    todayPLL = []
    (_, nt) = prcHist.shape
    for t in range(start, start+251):
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

        for pair in pairs:
            t1, t2 = pair
            key = f"{t1}-{t2}"
            dvol = np.sum(deltaPos[[t1,t2]])
            comm = dvol * commRate
            dcash = comm + curPrices[[t1,t2]].dot(deltaPos[[t1,t2]])
            pnls[key].append(pnls[key][-1] - dcash)
            positions[key].append((newPos[t1], newPos[t2]))

        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)
        todayPL = cash + posValue - value
        todayPLL.append(todayPL)
        value = cash + posValue
        ret = 0.0
        if (totDVolume > 0):
            ret = value / totDVolume

        prices.append(curPrices[ticker])
        values.append(value)
        volumes.append(curPos[ticker])


        # print("Day %d value: %.2lf todayPL: $%.2lf $-traded: %.0lf return: %.5lf" %
        #       (t, value, todayPL, totDVolume, ret))
    pll = np.array(todayPLL)
    (plmu, plstd) = (np.mean(pll), np.std(pll))
    annSharpe = 0.0
    if (plstd > 0):
        annSharpe = np.sqrt(250) * plmu / plstd
    return (plmu, ret, plstd, annSharpe, totDVolume)


(meanpl, ret, plstd, sharpe, dvol) = calcPL(prcAll)
score = meanpl - 0.1*plstd
print("=====")
print(f'start {start}')
print("mean(PL): %.1lf" % meanpl)
print("return: %.5lf" % ret)
print("StdDev(PL): %.2lf" % plstd)
print("annSharpe(PL): %.2lf " % sharpe)
print("totDvolume: %.0lf " % dvol)
print("Score: %.2lf" % score)

fig, ax = plt.subplots(len(pairs)+1,2, figsize=(20,len(pairs)*4))
fig.tight_layout()
ax[0,0].plot(values)

for i,p in enumerate(pairs):
    t1,t2 = p
    key = f"{t1}-{t2}"
    ax[i+1,0].title.set_text(key)
    ax[i+1,0].plot(pnls[key], label=key)
    ax[i+1,1].plot(positions[key], label=key)

# ax[1].plot(prices)
# ax[2].plot(volumes)
# plt.show()
fig.savefig('out.png')