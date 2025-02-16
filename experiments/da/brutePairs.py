import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import math

import threading

# Import for cointegration
from arch.unitroot.cointegration import engle_granger

import warnings
warnings.filterwarnings("ignore")

# Check if stationary
# Non-stationary
def test_stationary(series, printing=False, threshold=0.07):
    dftest = stattools.adfuller(series, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    if printing:
        print("Results of Dickey-Fuller Test:")
        print(dfoutput)

    
    kpsstest = stattools.kpss(series, regression="c", nlags="auto")
    kpss_output = pd.Series(
        kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
    )
    for key, value in kpsstest[3].items():
        kpss_output["Critical Value (%s)" % key] = value
    if printing:
        print("Results of KPSS Test:")
        print(kpss_output)
    
    isStationary = dfoutput["p-value"] < threshold or kpss_output["p-value"] > threshold
    if printing:
        print("STATIONARY: ", isStationary)    

    return (dfoutput["p-value"], kpss_output["p-value"], isStationary)

# Count number of times crossing the mean
def countCrossingFreq(df, mean):
    count = 0
    for i in range(1, len(df)):
        if (df[i] - mean) * (df[i-1] - mean) < 0:
            count += 1
    return count

def testCrossingFreq(deltaTrain, deltaTest, mean):
    freqTrain = countCrossingFreq(deltaTrain, mean)
    freqTest = countCrossingFreq(deltaTest, mean)
    # print(freqTrain)
    # print(freqTest)
    return ((freqTest/freqTrain > 0.6 * 3/7 and freqTest > 10 and freqTrain > 30), freqTrain, freqTest)

# Test if it's actually mean reverting by fitting a linear regression
HORIZONTAL_THRESHOLD = 0.006
def testHorizontal(deltaFull):
    x = list(range(len(deltaFull)))
    y = list(deltaFull)
    m, b = np.polyfit(x, y, 1)

    checkFull = abs(m) < HORIZONTAL_THRESHOLD
    checkSegment = True

    for start in range(0, len(deltaFull) - 500, 250):
        m, b = np.polyfit(x[start:start + 500], y[start:start+500], 1)
        checkSegment = checkSegment and abs(m) < HORIZONTAL_THRESHOLD

    return checkFull and checkSegment 

def totalTest(pair, curBeta):
    deltaTrain = train[str(pair[0])] + train[str(pair[1])] * curBeta
    deltaTest = test[str(pair[0])] + test[str(pair[1])] * curBeta
    deltaFull = df[str(pair[0])] + df[str(pair[1])] * curBeta

    # The stationary test is a bit too strong. Might check if want to add.
    # checkStationary = (test_stationary(deltaTrain, printing=True)[2] 
    #     and test_stationary(deltaTest, printing=True)[2] 
    #     and test_stationary(deltaFull, printing=True)[2])
    checkCrossingFreq, freqTrain, freqTest = testCrossingFreq(deltaTrain, deltaTest, np.mean(deltaFull))
    checkHorizontal = testHorizontal(deltaFull)
    return (checkHorizontal, checkCrossingFreq, freqTrain, freqTest)

def bruteStep(pair: tuple, result: list):
    maxCross = 0
    maxBeta = 0

    for step in range(-300, 301, 1):
        curBeta = step/100
        checkCrossingFreq, checkHorizontal, freqTrain, freqTest = totalTest(pair, curBeta)

        if checkCrossingFreq and checkHorizontal:
            if (freqTrain + freqTest > maxCross):
                maxCross = freqTest + freqTrain
                maxBeta = curBeta

    if maxCross > 60:
        result.append((pair, (1, maxBeta), maxCross))
    print(result)

def bruteStepMultiple(pairs: list, result: list):
    if len(pairs) == 0:
        return
    for pair in pairs:
        bruteStep(pair, result)

df = pd.read_csv("../prices.txt", sep='\s+', header=None, index_col=None)
df.rename(columns=lambda c: str(c), inplace=True)
df["time"] = pd.to_datetime([i for i in range(1000)], unit='D', origin=pd.Timestamp('2020-01-01'))
df.set_index("time", inplace=True)
# df = np.log(df)


train = df[:700][:]
test = df[700:1000][:]


# # Find the coefficient that generalizes well 
# result = []
# allPairs = [(i, j) for i in range(50) for j in range(i+1, 50)]

# # There are 12 cores, so sprung up 8 threads
# threads = []
# segment = math.ceil(len(allPairs)/8)
# print("Each thread has: ", segment)

# # Create threads
# for i in range(0, len(allPairs), segment):
#     threads.append(threading.Thread(target=bruteStepMultiple, args=(allPairs[i:i+segment], result)))

# # Start all of them
# for thread in threads:
#     thread.start()

# # Wait until all of them finishes
# for thread in threads:
#     thread.join()

# print("Done")
# print(result) 


result = [((24, 49), (1, -1.76), 135), ((3, 14), (1, -1.94), 77), ((14, 38), (1, -1.01), 83), ((0, 4), (1, -0.82), 66), ((14, 39), (1, -1.05), 79), ((6, 40), (1, -0.46), 62), ((6, 41), (1, 0.1), 64), ((32, 44), (1, -1.28), 61), ((14, 41), (1, 0.13), 71), ((0, 7), (1, 0.39), 65), ((14, 42), (1, -1.08), 71), ((14, 44), (1, -0.34), 117), ((0, 9), (1, 0.41), 81), ((14, 45), (1, -0.56), 90), ((32, 49), (1, -2.71), 79), ((14, 46), (1, 0.12), 75), ((19, 40), (1, -3.0), 67), ((0, 13), (1, -0.67), 70), ((14, 49), (1, -0.74), 109), ((0, 14), (1, -1.33), 72), ((15, 17), (1, 2.04), 64), ((0, 16), (1, -1.87), 65), ((19, 46), (1, 0.3), 73), ((3, 34), (1, -1.07), 103), ((3, 36), (1, 0.58), 63), ((11, 13), (1, -1.05), 63), ((11, 14), (1, -2.73), 73), ((20, 25), (1, -0.49), 68), ((0, 27), (1, -1.9), 63), ((3, 41), (1, 0.29), 71), ((0, 29), (1, -1.6), 62), ((34, 40), (1, -1.8), 66), ((0, 32), (1, -0.57), 71), ((11, 23), (1, -1.25), 61), ((20, 35), (1, -0.64), 72), ((15, 37), (1, -1.32), 63), ((4, 5), (1, -1.55), 62), ((4, 7), (1, 0.38), 72), ((0, 40), (1, -1.19), 75), ((15, 43), (1, -0.17), 63), ((11, 33), (1, -0.92), 73), ((11, 34), (1, -0.71), 69), ((15, 47), (1, 2.33), 61), ((4, 15), (1, -1.27), 70), ((11, 36), (1, 0.46), 65), ((0, 49), (1, -1.33), 80), ((21, 23), (1, -0.25), 61), ((11, 41), (1, 0.33), 65), ((1, 3), (1, -1.18), 63), ((16, 21), (1, -1.3), 74), ((11, 42), (1, -2.21), 96), ((21, 27), (1, -0.71), 70), ((16, 24), (1, -0.52), 74), ((21, 28), (1, -0.42), 65), ((21, 31), (1, -0.14), 65), ((36, 41), (1, -0.36), 101), ((12, 13), (1, -0.12), 71), ((12, 14), (1, -0.18), 91), ((8, 15), (1, -0.36), 62), ((12, 15), (1, -0.07), 63), ((4, 32), (1, -0.67), 74), ((12, 16), (1, -0.27), 67), ((8, 17), (1, 1.24), 78), ((12, 18), (1, -0.27), 85), ((21, 40), (1, -0.53), 69), ((12, 20), (1, 0.12), 92), ((4, 38), (1, -2.33), 70), ((12, 22), (1, -0.04), 79), ((21, 43), (1, -0.27), 72), ((12, 23), (1, -0.11), 69), ((12, 24), (1, -0.09), 75), ((12, 25), (1, 0.05), 72), ((12, 27), (1, -0.34), 71), ((4, 44), (1, -0.64), 72), ((12, 28), (1, -0.48), 63), ((4, 45), (1, -1.21), 67), ((12, 29), (1, -0.16), 91), ((12, 30), (1, -0.42), 72), ((37, 47), (1, 2.17), 96), ((8, 32), (1, -0.13), 68), ((12, 31), (1, -0.16), 67), ((22, 24), (1, -2.46), 86), ((4, 49), (1, -1.81), 86), ((12, 34), (1, -0.17), 73), ((5, 7), (1, 0.4), 67), ((12, 35), (1, 0.04), 79), ((1, 34), (1, -1.6), 65), ((8, 37), (1, -0.68), 74), ((12, 36), (1, 0.08), 91), ((5, 9), (1, 0.5), 62), ((8, 39), (1, -0.17), 68), ((12, 38), (1, -0.22), 71), ((12, 39), (1, -0.19), 69), ((5, 13), (1, -0.68), 71), ((12, 41), (1, 0.04), 85), ((5, 14), (1, -1.74), 93), ((8, 43), (1, -0.27), 64), ((12, 42), (1, -0.42), 87), ((1, 41), (1, 0.38), 73), ((38, 49), (1, -1.03), 99), ((12, 44), (1, -0.08), 63), ((12, 46), (1, 0.03), 75), ((39, 43), (1, -0.33), 62), ((39, 44), (1, -0.29), 71), ((12, 49), (1, -0.26), 87), ((13, 14), (1, -2.84), 111), ((2, 3), (1, -0.32), 63), ((5, 24), (1, -0.84), 64), ((9, 13), (1, 1.16), 79), ((39, 49), (1, -1.09), 82), ((13, 18), (1, -2.22), 73), ((13, 19), (1, -0.9), 63), ((2, 7), (1, 0.25), 65), ((5, 28), (1, -2.85), 83), ((9, 18), (1, 2.86), 78), ((2, 9), (1, 0.3), 64), ((2, 11), (1, -0.53), 61), ((5, 32), (1, -0.54), 92), ((2, 12), (1, -2.32), 73), ((2, 13), (1, -0.32), 63), ((2, 14), (1, -1.04), 80), ((40, 49), (1, -1.23), 77), ((2, 16), (1, -0.76), 65), ((5, 38), (1, -1.89), 81), ((9, 29), (1, 2.9), 78), ((2, 19), (1, -0.57), 63), ((13, 32), (1, -0.95), 83), ((2, 21), (1, -1.41), 64), ((23, 36), (1, 0.6), 94), ((13, 36), (1, 0.61), 72), ((5, 46), (1, 0.17), 77), ((9, 36), (1, -1.11), 94), ((13, 38), (1, -2.92), 77), ((5, 49), (1, -1.4), 101), ((13, 40), (1, -2.56), 71), ((6, 7), (1, 0.09), 81), ((9, 40), (1, 3.0), 77), ((42, 49), (1, -1.07), 73), ((6, 8), (1, -0.53), 64), ((13, 44), (1, -0.75), 80), ((13, 45), (1, -1.39), 82), ((9, 44), (1, 1.61), 68), ((13, 46), (1, 0.24), 75), ((9, 46), (1, -0.42), 66), ((44, 45), (1, -1.91), 77), ((14, 16), (1, -1.21), 102), ((14, 18), (1, -0.9), 91), ((6, 19), (1, -0.35), 67), ((44, 49), (1, -1.82), 61), ((14, 19), (1, -0.57), 63), ((14, 21), (1, -1.7), 62), ((45, 49), (1, -1.18), 67), ((14, 23), (1, -0.27), 63), ((14, 24), (1, -0.49), 70), ((14, 27), (1, -1.11), 116), ((6, 28), (1, -1.2), 81), ((47, 49), (1, 0.46), 75), ((6, 29), (1, -0.43), 67), ((14, 29), (1, -1.17), 72), ((6, 30), (1, -1.32), 62), ((14, 30), (1, -2.19), 78), ((14, 32), (1, -0.3), 81), ((14, 34), (1, -0.67), 79)]
result.sort(key=lambda x: -x[2])
print(result)