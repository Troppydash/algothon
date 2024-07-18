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




df = pd.read_csv("../prices.txt", sep='\s+', header=None, index_col=None)
df.rename(columns=lambda c: str(c), inplace=True)
df["time"] = pd.to_datetime([i for i in range(1000)], unit='D', origin=pd.Timestamp('2020-01-01'))
df.set_index("time", inplace=True)
df = np.log(df)


train = df[:700][:]
test = df[700:1000][:]


# Find the coefficient that generalizes well 
result = []

for i in range(50):
    for j in range(i+1, 50):
        pair = (i, j)
        maxCross = 0
        maxBeta = 0

        for step in range(-300, 301, 1):
            curBeta = step/100
            deltaTrain = train[str(pair[0])] + train[str(pair[1])] * curBeta
            deltaTest = test[str(pair[0])] + test[str(pair[1])] * curBeta
            deltaFull = df[str(pair[0])] + df[str(pair[1])] * curBeta

            # The stationary test is a bit too strong. Might check if want to add.
            # checkStationary = (test_stationary(deltaTrain, printing=True)[2] 
            #     and test_stationary(deltaTest, printing=True)[2] 
            #     and test_stationary(deltaFull, printing=True)[2])
            checkCrossingFreq, freqTrain, freqTest = testCrossingFreq(deltaTrain, deltaTest, np.mean(deltaFull))
            checkHorizontal = testHorizontal(deltaFull)

            # print(checkCrossingFreq, checkHorizontal)

            if checkCrossingFreq and checkHorizontal:
                if (freqTrain + freqTest > maxCross):
                    maxCross = freqTest + freqTrain
                    maxBeta = curBeta

        if maxCross > 60:
            result.append((pair, (1, maxBeta), maxCross))
        print(result)

print(result) 


