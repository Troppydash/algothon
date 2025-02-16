import random
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.tsa.ardl as ardl
import statsmodels.tsa.arima.model as arima
import statsmodels.regression as reg
import arch

if False:
    from arbitragelab.copula_approach import fit_copula_to_empirical_data
    from arbitragelab.copula_approach.elliptical import GaussianCopula
    from arbitragelab.trading import BasicCopulaTradingRule


# test_t1 = 7
# test_t2 = 19

# fitting
def fitting(test_t1, test_t2):
    df = pd.read_csv("./prices.txt", sep='\s+', header=None, index_col=None)
    df.index = np.arange(df.shape[0])
    df.rename(columns=lambda c: str(c), inplace=True)

    train = df.iloc[:300, :]

    cop_trading = BasicCopulaTradingRule(exit_rule='or', open_probabilities=(0.15, 0.85),
                                         exit_probabilities=(0.5, 0.5))

    # Split data into train and test sets

    # Fitting copula to data and getting cdf for X and Y series
    info_crit, fit_copula, ecdf_x, ecdf_y = fit_copula_to_empirical_data(x=train[str(test_t1)],
                                                                         y=train[str(test_t2)],
                                                                         copula=GaussianCopula)

    # Setting initial probabilities
    cop_trading.current_probabilities = (0.5, 0.5)
    cop_trading.prev_probabilities = (0.5, 0.5)

    # Adding copula to strategy
    cop_trading.set_copula(fit_copula)

    # Adding cdf for X and Y to strategy
    cop_trading.set_cdf(ecdf_x, ecdf_y)

    return cop_trading


def copula(df, test_t1, test_t2, trading, p1=0.43, p2=-0.56, num=450, sp=1.5):
    trading.update_probabilities(df[test_t1].values[-1], df[test_t2].values[-1])

    trade, side = trading.check_entry_signal()
    result = trading.update_trades(update_timestamp=pd.Timestamp(iter))
    for i in result:
        # print(f'closing {i}')
        amount[f"copula-1-{test_t1}-{test_t2}-{i}"] = (test_t1, 0)
        amount[f"copula-2-{test_t1}-{test_t2}-{i}"] = (test_t2, 0)

    if trade:
        total = aggregate()
        NUM = num
        # power1 = (NUM - abs(total[test_t1])) / sp
        power1 = NUM / 15
        power2 = NUM / 15
        # power2 = (NUM - abs(total[test_t2])) / sp
        amount[f"copula-1-{test_t1}-{test_t2}-{iter}"] = (test_t1, int(side * power1 * p1))
        amount[f"copula-2-{test_t1}-{test_t2}-{iter}"] = (test_t2, int(side * power2 * -p2))
        # print(f'opening {iter}')
        trading.add_trade(start_timestamp=pd.Timestamp(iter), side_prediction=side, uuid=iter)


##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

currentPos = np.zeros(50)


def set_value(df, ticker, value):
    value = clamp(value, -10000, 10000)
    val = df[ticker].values[-1]
    vol = int(value / val)

    assert -10000 <= vol * val <= 10000
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


last = defaultdict(lambda: 0)
amount = defaultdict(lambda: (None, 0))


def better_pair(df, t1, t2, beta, lower, middle, upper, s1, s2, v1, v2, scaler: float = 1.1):
    global last

    delta = df[t1].values[-1] - beta * df[t2].values[-1]

    if delta < lower and last[t1, t2] != -1:
        last[t1, t2] = -1
        amount[f"{t1}-{t2}-1"] = (t1, int(s1 * v1 * scaler))
        amount[f"{t1}-{t2}-2"] = (t2, -int(s2 * v2 * scaler))

    # removing this is somehow better?
    # if last[t1, t2] < 0 and delta >= middle or last[t1, t2] > 0 and delta <= middle:
    #     last[t1, t2] = 0
    #     amount[f"{t1}-{t2}-1"] = (t1, int(amount[f"{t1}-{t2}-1"][1] // 1.5))
    #     amount[f"{t1}-{t2}-2"] = (t2, int(amount[f"{t1}-{t2}-2"][1] // 1.5))

    if delta > upper and last[t1, t2] != 1:
        last[t1, t2] = 1
        amount[f"{t1}-{t2}-1"] = (t1, -int(s1 * v1 * scaler))
        amount[f"{t1}-{t2}-2"] = (t2, int(s2 * v2 * scaler))


def better_one(df, t1, lower, middle, upper, s1, v1, scaler: float = 1.1):
    global last

    delta = df[t1].values[-1]

    if delta < lower and last[t1, -1] != -1:
        last[t1, -1] = -1
        currentPos[t1] = int(s1 * v1 * scaler)
    #
    if last[t1, -1] < 0 and delta >= middle or last[t1, -1] > 0 and delta <= middle:
        last[t1, -1] = 0
        currentPos[t1] = currentPos[t1] // 1.5

    if delta > upper and last[t1, -1] != 1:
        last[t1, -1] = 1
        currentPos[t1] = -int(s1 * v1 * scaler)


def regularize(df):
    for i in range(50):
        pos, val = currentPos[i], df[i].values[-1]
        if pos * val > 10000:
            currentPos[i] = int(10000 / val)
        if pos * val < -10000:
            currentPos[i] = int(-10000 / val)


def aggregate():
    for key, value in amount.items():
        currentPos[value[0]] = 0

    for key, value in amount.items():
        currentPos[value[0]] += value[1]


if False:
    trading = fitting(7, 19)
    trading2 = fitting(11, 34)
    trading3 = fitting(18, 36)
    trading4 = fitting(3, 36)
iter = 0


def predict(df, ticker, indices, deg):
    result = ardl.ARDL(
        df[ticker].pct_change().dropna().values[-200:],
        0,
        df[indices].pct_change().dropna().values[-200:, :],
        [1],
        causal=True
    ).fit()

    predict_mu = result.forecast()
    est = predict_mu
    val = df[ticker].values[-1]
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    pforecast = est * val
    if abs(vol * pforecast) > trans:
        if pforecast > 0:
            currentPos[ticker] = vol
        else:
            currentPos[ticker] = -vol


number = 0


def predict_one(df, ticker, deg):
    import warnings
    warnings.filterwarnings("ignore")

    result = arima.ARIMA(
        df[ticker].pct_change().dropna().values[-250:],
        order=(3, 0, 3)
    ).fit()

    predict_mu = result.forecast()
    est = predict_mu
    val = df[ticker].values[-1]
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    pforecast = est * val
    # print(est*val*vol)
    if abs(vol * pforecast) > trans:
        if pforecast > 0:
            currentPos[ticker] = vol
        else:
            currentPos[ticker] = -vol


def getMyPosition(prices):
    global currentPos, iter

    iter += 1

    nins, nt = prices.shape
    if nt < 200:
        return np.zeros(nins)

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    if True:
        predict(df, 38, list(range(50)), 1.1)
        predict(df, 27, list(range(50)), 1.1)

    if True:
        better_pair(df, 28, 39, 1.2159226859939878, -8.963364425168958, -8.783364425168958,
                    -8.603364425168959,
                    7.0, 8.0, 27.414535186555913, 25.145845906256287, 0.6)
        better_pair(df, 11, 42, 2.81, -9.48586161, -8.89586161, -8.30586161, 8, 21, 37, 32, 0.6)
        better_pair(df, 12, 34, 0.23829431834048995, 20.657766061851223, 20.757766061851225,
                    20.857766061851226,
                    9.0, 2.0, 40.24306813151435, 175.13134851138352, 0.6)
        better_pair(df, 14, 30, 0.7192064351085297, 4.86640231, 5.15640231, 5.44640231, 6, 4, 98, 162)
        better_pair(df, 22, 24, 3.4117759223360125, -149.87702011, -148.68702011, -147.49702011, 8, 27, 17, 5)
        better_pair(df, 28, 41, 0.021148, 49.6408878, 49.8208878, 50.0008878, 48, 1, 3, 129, 0.6)
        better_pair(df, 7, 19, 0.3973201339866572, 33.5724356, 34.1524356, 34.7324356, 13, 5, 15, 56, 0.3)
        better_pair(df, 43, 49, 0.2046650849773665, 47.71489042555398, 48.64489042555398, 49.57489042555398,
                    10.0,
                    2.0, 15.130882130428203, 84.30281571404484)
        better_pair(df, 12, 25, -0.06366267571277003, 29.863092873692395, 29.983092873692396,
                    30.103092873692397,
                    16.0, 1.0, 22.63672582397682, 140.66676044450696, 0.6)
        better_pair(df, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299, 39.807363604822996,
                    12.0,
                    2.0, 16.311085013375088, 99.96001599360255, 0.8)
        better_pair(df, 2, 11, 0.22442595136426555, 41.245986043711305, 41.355986043711304,
                    41.465986043711304, 9.0, 2.0, 22.857665318064413, 149.1201908738443, 0.6)
        better_one(df, 8, 68.08943333, 68.26943333, 68.44943333, 1, 10000 // 80)

    if False:
        copula(df, 7, 19, trading, 0.43, 0.56, 400)
        copula(df, 11, 34, trading2, 0.46, 0.53, 650)
        copula(df, 18, 36, trading3, 1.83, -0.83, 200)
        # copula(df, 12, 16, trading3, 0.59, 0.40, 550, 1.5)
        # copula(df, 27, 47, trading4, 0.56, 0.44, 550)
        # copula(df, 3, 36, trading4, 0.43, 0.56, 300)

    aggregate()
    regularize(df)

    return currentPos


if __name__ == "__main__":
    from custom_eval.alleval import all_eval

    all_eval(getMyPosition)
