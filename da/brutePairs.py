import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import math

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
def testHorizontal(deltaFull):
    x = list(range(len(deltaFull)))
    y = list(deltaFull)
    m, b = np.polyfit(x, y, 1)

    checkFull = abs(m) < 0.0002
    checkSegment = True

    for start in range(0, len(deltaFull) - 500, 250):
        m, b = np.polyfit(x[start:start + 500], y[start:start+500], 1)
        checkSegment = checkSegment and abs(m) < 0.0002

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


df = pd.read_csv("../prices.txt", sep='\s+', header=None, index_col=None)
df.rename(columns=lambda c: str(c), inplace=True)
df["time"] = pd.to_datetime([i for i in range(1000)], unit='D', origin=pd.Timestamp('2020-01-01'))
df.set_index("time", inplace=True)
df = np.log(df)


train = df[:700][:]
test = df[700:1000][:]


# # Find the coefficient that generalizes well 
# result = []

# for i in range(50):
#     for j in range(i+1, 50):
#         pair = (i, j)
#         maxCross = 0
#         maxBeta = 0

#         for step in range(-300, 301, 1):
#             curBeta = step/100
#             checkCrossingFreq, checkHorizontal, freqTrain, freqTest = totalTest(pair, curBeta)

#             if checkCrossingFreq and checkHorizontal:
#                 if (freqTrain + freqTest > maxCross):
#                     maxCross = freqTest + freqTrain
#                     maxBeta = curBeta

#         if maxCross > 60:
#             result.append((pair, (1, maxBeta), maxCross))
#         print(result)

# print(result) 


# result = [((9, 13), (1, 1.0), 77), ((3, 9), (1, 0.81), 68), ((45, 49), (1, -1.97), 77), ((13, 29), (1, -1.82), 65), ((28, 38), (1, -0.21), 81), ((6, 8 ), (1, -1.64), 70), ((16, 49), (1, -1.88), 103), ((28, 43), (1, 0.34), 72), ((39, 43), (1, -0.39), 62), ((20, 25), (1, -0.49), 77), ((24, 46), (1, 0.24), 70), ((27, 45), (1, -0.65), 68), ((37, 49), (1, -0.24), 76), ((1, 12), (1, -2.54), 87), ((2, 45), (1, -0.42), 74), ((3, 30), (1, -1.45), 62), ((24, 28), (1, -2.88), 109), ((32, 49), (1, -2.96), 87), ((19, 40), (1, -3.0), 61), ((12, 49), (1, -0.57), 89), ((1, 2), (1, -2.4), 65), ((30, 36), (1, 0.58), 81), ((13, 41), (1, 0.28), 68), ((21, 34), (1, -0.3), 71), ((1, 36), (1, 0.65), 74), ((29, 49), (1, -1.23), 92), ((11, 23), (1, -1.4), 65), ((24, 49), (1, -1.69), 129), ((13, 18), (1, -0.61), 75), ((29, 46), (1, 0.27), 74), ((18, 40), (1, -1.82), 61), ((16, 21), (1, -0.83), 74), ((38, 45), (1, -1.2), 62), ((0, 32), (1, -2.36), 66), ((13, 45), (1, -1.49), 84), ((24, 27), (1, -1.01), 72), ((28, 33), (1, -0.04), 61), ((30, 42), (1, -0.46), 84), ((44, 49), (1, -2.98), 61), ((14, 45), (1, -2.07), 100), ((2, 32), (1, -0.4), 65), ((6, 7), (1, 0.25), 69), ((9, 30), (1, 1.06), 78), ((13, 38), (1, -1.12), 74), ((21, 22), (1, -0.4), 65), ((12, 35), (1, 0.11), 83), ((12, 25), (1, 0.13), 78), ((14, 44), (1, -0.84), 121), ((2, 38), (1, -0.29), 67), ((17, 43), (1, 0.11), 90), ((18, 34), (1, -0.89), 66), ((24, 38), (1, -0.67), 66), ((11, 14), (1, -1.51), 69), ((28, 40), (1, -0.22), 77), ((28, 37), (1, -0.29), 92), ((16, 40), (1, -0.56), 84), ((9, 36), (1, -0.77), 92), ((12, 28), (1, -0.95), 63), ((29, 36), (1, 0.41), 65), ((27, 28), (1, -2.84), 77), ((13, 42), (1, -0.74), 63), ((9, 46), (1, -0.52), 70), ((11, 41), (1, 0.57), 69), ((0, 7), (1, 1.97), 81), ((40, 49), (1, -2.18), 81), ((13, 36), (1, 0.37), 70), ((4, 32), (1, -0.98), 77), ((2, 12), (1, -1.7), 76), ((22, 24), (1, -2.36), 86), ((22, 49), (1, -2.97), 89), ((44, 45), (1, -2.97), 69), ((12, 36), (1, 0.12), 91), ((1, 41), (1, 0.4), 77), ((9, 27), (1, 2.17), 67), ((16, 24), (1, -1.02), 64), ((4, 45), (1, -1.15), 69), ((19, 38), (1, -1.72), 73), ((3, 41), (1, 0.51), 68), ((12, 44), (1, -0.1), 63), ((39, 44), (1, -0.2), 61), ((19, 29), (1, -2.76), 61), ((4, 5), (1, -0.24), 70), ((36, 41), (1, -0.65), 95), ((9, 42), (1, 0.68), 84), ((2, 28), (1, -2.5), 77), ((1, 46), (1, 0.36), 67), ((16, 27), (1, -0.81), 68), ((12, 13), (1, -0.15), 69), ((20, 35), (1, -0.41), 98), ((5, 32), (1, -2.62), 84), ((17, 26), (1, 0.09), 98), ((28, 30), (1, -0.34), 77), ((9, 11), (1, 0.79), 91), ((13, 39), (1, -2.83), 72), ((34, 40), (1, -2.45), 66), ((21, 43), (1, -0.72), 76), ((15, 22), (1, -0.68), 61), ((12, 24), (1, -0.2), 77), ((7, 49), (1, 2.92), 62), ((29, 34), (1, -0.32), 63), ((1, 35), (1, 0.43), 61), ((14, 21), (1, -2.87), 64), ((3, 16), (1, -1.65), 81), ((23, 30), (1, -1.43), 62), ((2, 14), (1, -0.33), 85), ((2, 22), (1, -0.34), 62), ((4, 7), (1, 0.34), 70), ((31, 47), (1, 1.18), 76), ((3, 34), (1, -0.53), 104), ((14, 42), (1, -0.89), 71), ((9, 18), (1, 0.66), 78), ((12, 30), (1, -0.96), 68), ((14, 38), (1, -2.22), 72), ((8, 39), (1, -0.12), 66), ((19, 46), (1, 0.77), 74), ((14, 19), (1, -0.91), 70), ((4, 44), (1, -0.48), 62), ((12, 22), (1, -0.09), 89), ((30, 33), (1, -0.39), 72), ((7, 44), (1, 0.91), 86), ((28, 34), (1, -0.1), 69), ((27, 32), (1, -0.53), 71), ((12, 46), (1, 0.08), 77), ((13, 14), (1, -0.88), 109), ((28, 36), (1, 0.1), 65), ((28, 45), (1, -0.21), 77), ((30, 40), (1, -0.81), 68), ((14, 27), (1, -2.1), 97), ((3, 14), (1, -0.57), 81), ((18, 29), (1, -2.51), 74), ((37, 39), (1, -0.59), 70), ((2, 49), (1, -1.1), 95), ((39, 49), (1, -1.31), 83), ((3, 36), (1, 0.51), 69), ((6, 29), (1, -0.73), 65), ((17, 25), (1, 0.04), 70), ((28, 32), (1, -0.16), 79), ((29, 32), (1, -0.56), 75), ((4, 14), (1, -0.48), 62), ((11, 42), (1, -0.89), 97), ((16, 28), (1, -2.97), 85), ((7, 32), (1, 1.31), 62), ((1, 3), (1, -0.8 ), 62), ((32, 38), (1, -1.39), 68), ((28, 47), (1, 0.23), 61), ((14, 34), (1, -0.8 ), 75), ((9, 23), (1, 0.52), 69), ((12, 27), (1, -0.41), 72), ((27, 49), (1, -1.79), 113), ((12, 41), (1, 0.08), 81), ((13, 23), (1, -1.05), 65), ((28, 46), (1, 0.06), 75), ((6, 18), (1, -0.23), 67), ((7, 45), (1, 2.08), 67), ((7, 16), (1, 2.68), 102), ((1, 30), (1, -0.62), 81), ((6, 30), (1, -0.92), 62), ((30, 41), (1, 0.2), 74), ((4, 13), (1, -0.73), 74), ((12, 42), (1, -0.2), 79), ((2, 19), (1, -0.36), 69), ((8, 17), (1, 0.81), 78), ((17, 20), (1, 0.09), 62), ((31, 37), (1, -0.54), 75), ((11, 13), (1, -1.98), 61), ((4, 49), (1, -2.17), 90), ((6, 22), (1, -0.41), 63), ((4, 46), (1, 0.3), 64), ((13, 34), (1, -0.7), 67), ((4, 38), (1, -0.92), 66), ((43, 44), (1, -0.48), 62), ((16, 38), (1, -0.69), 63), ((23, 36), (1, 0.82), 85), ((12, 29), (1, -0.18), 89), ((1, 18), (1, -0.61), 61), ((14, 16), (1, -2.86), 102), ((11, 33), (1, -1.39), 69), ((8, 37), (1, -0.7), 76), ((2, 11), (1, -0.25), 64), ((30, 46), (1, 0.24), 62), ((12, 38), (1, -0.17), 69), ((7, 21), (1, 2.02), 65), ((8, 28), (1, -1.57), 81), ((12, 39), (1, -0.35), 71), ((13, 44), (1, -0.53), 72), ((7, 38), (1, 1.81), 85), ((9, 40), (1, 1.14), 72), ((13, 46), (1, 0.39), 77), ((17, 37), (1, 0.07), 70), ((12, 19), (1, -0.1), 63), ((9, 34), (1, 0.91), 61), ((9, 16), (1, 2.62), 63), ((9, 14), (1, 1.06), 77), ((10, 37), (1, 1.31), 80), ((27, 29), (1, -0.98), 68), ((28, 49), (1, -0.86), 99), ((29, 30), (1, -0.82), 68), ((2, 7), (1, 0.28), 61), ((13, 19), (1, -0.52), 65), ((9, 41), (1, -0.46), 70), ((13, 27), (1, -2.27), 89), ((1, 14), (1, -0.97), 65), ((12, 18), (1, -0.13), 85), ((11, 30), (1, -2.23), 64), ((28, 44), (1, -0.11), 77), ((12, 23), (1, -0.14), 66), ((28, 39), (1, -0.39), 71), ((1, 34), (1, -0.54), 61), ((12, 14), (1, -0.09), 85), ((28, 31), (1, -0.44), 97), ((40, 46), (1, 0.31), 62), ((9, 29), (1, 2.01), 76), ((14, 18), (1, -0.85), 93), ((13, 16), (1, -2.08), 61), ((8, 32), (1, -0.1), 62), ((12, 34), (1, -0.15), 75), ((37, 47), (1, 2.02), 93), ((21, 40), (1, -0.74), 63), ((13, 32), (1, -1.08), 81), ((2, 13), (1, -0.36), 65), ((47, 49), (1, 0.73), 73), ((12, 16), (1, -0.36), 65), ((2, 16), (1, -0.57), 61), ((4, 47), (1, 2.56), 63), ((38, 49), (1, -2.23), 99), ((4, 15), (1, -0.87), 95), ((14, 46), (1, 0.62), 73), ((13, 49), (1, -1.92), 74), ((12, 20), (1, 0.28), 78), ((24, 40), (1, -0.74), 77)]
# result.sort(key=lambda x: -x[2])
# print(result[:10])

print(totalTest((24, 49), -1.69))