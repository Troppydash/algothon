from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.tsa.ardl as ardl
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
    #
    if last[t1, t2] < 0 and delta >= middle or last[t1, t2] > 0 and delta <= middle:
        last[t1, t2] = 0
        amount[f"{t1}-{t2}-1"] = (t1, int(amount[f"{t1}-{t2}-1"][1] // 1.5))
        amount[f"{t1}-{t2}-2"] = (t2, int(amount[f"{t1}-{t2}-2"][1] // 1.5))

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


def predict_raw(df, ticker, deg):
    result = ardl.ARDL(
        df[ticker].values[-100:],
        1,
        df.values[-100:, :],
        1,
        causal=True
    ).fit()

    predict_mu = result.forecast()
    est = predict_mu
    val = df[ticker].values[-1]
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    pforecast = (est - val)
    if abs(vol * pforecast) > trans:
        if pforecast > 0:
            currentPos[ticker] = vol
        else:
            currentPos[ticker] = -vol


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


def getMyPosition(prices):
    global currentPos, iter

    iter += 1

    nins, nt = prices.shape
    if nt < 250:
        return np.zeros(nins)

    df = pd.DataFrame(prices.T, columns=np.arange(50))
    weights = [-0.1344368, -0.035686, -0.1482424, -0.1665422, -0.0748774, 0.9619431]
    indices = [5, 13, 16, 28, 30, 38]
    # best_pair(df, weights, indices, 2.586489376879487, 2.6713907573920013, 2.7562921379045155)

    used = {28, 39, 11, 42, 12, 34, 14, 30, 22, 24, 28, 41, 7, 19, 43, 49, 13, 39, 12, 25, 7, 48}

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
        better_pair(df, 13, 39, -2.935182925012027, 193.2674924878151, 193.84749248781512, 194.42749248781513,
                    7.0, 19.0, 27.78240817914097, 10.587724592107909)
        better_pair(df, 12, 25, -0.06366267571277003, 29.863092873692395, 29.983092873692396,
                    30.103092873692397,
                    16.0, 1.0, 22.63672582397682, 140.66676044450696, 0.6)
        better_pair(df, 7, 48, 0.1804261347577773, 38.62736360482299, 39.21736360482299, 39.807363604822996,
                    12.0,
                    2.0, 16.311085013375088, 99.96001599360255, 0.8)
        better_pair(df, 2, 11, 0.22442595136426555, 41.245986043711305, 41.355986043711304,
                    41.465986043711304, 9.0, 2.0, 22.857665318064413, 149.1201908738443, 0.6)

        # better_one(df, 8, 68.08943333, 68.26943333, 68.44943333, 1, 10000 // 80)

    if False:
        copula(df, 7, 19, trading, 0.43, 0.56, 400)
        copula(df, 11, 34, trading2, 0.46, 0.53, 650)
        copula(df, 18, 36, trading3, 1.83, -0.83, 200)
        # copula(df, 12, 16, trading3, 0.59, 0.40, 550, 1.5)
        # copula(df, 27, 47, trading4, 0.56, 0.44, 550)
        # copula(df, 3, 36, trading4, 0.43, 0.56, 300)

    # 20,25 done
    # 21,34
    # 21,35
    # 24 41
    # 25 34
    # 27 47
    # 19 37
    # 18 36
    # 17 18
    # 16 25
    # 13 39
    # 12 34
    # 12 25
    # 12 15
    # 11 47
    # 11 20
    # 8 14
    # 7 48
    # 7 45
    # 7 32
    # 5 39
    # 4 47
    # 4 27
    # 4 11

    # better_pair(df, 5, 44, 0.15299458297216598, 6.19603119, 6.35603119, 6.51603119, 7, 1, 109, 255)
    # better_pair(df, currentPos[7]-currentPos[39], 7, 39, -1.1096567908341244, 100.64737749, 101.39737749, 102.14737749)
    # pair_trade_log(df, 28, 49, -0.09244661852645539, 0.018986374603144413, 2, 2.5, 9000)
    # pair_trade(df, 1, 10, 34.04245999999999, 1.603967876361618, 0.6, 2.5, 9000)
    # pair_trade_log(df, 7, 8, -0.38218823, 0.02883, 2, 2.5, 9000)
    # pair_trade_log(df, 43, 47, 0.5692941382086553, 0.03309117994281669, 1, 2.5, 9000)
    # pair_trade_log(df, 22, 24, 0.06450, 0.032329, 1, 2.5, 9000)
    # pair_trade_log(df, 14, 23, -0.697, 0.040451, 1, 2.5, 9000)
    # pair_trade_log(df, 28, 39, 0.03865, 0.005758, 1, 1.3, 9000)
    # another_pair(df, mr, 11, 42)
    # another_pair(df, mr2, 28, 39)
    # print((delta-m)/sd, pos)
    # z = (delta-m)/sd
    # if z > 0:
    #     print('11 > 42, short 11, buy 42')
    #     print(z, pos, currentPos[11], currentPos[42], mr.state)
    # else:
    #     print('11 < 42, buy 11, short 42')
    #     print(z, pos, currentPos[11], currentPos[42], mr.state)
    # pair_trade_log(df, 11, 42, 0.7556795088657547, 0.038963533345916526, 1, 2.5, 9000)

    # currentPos = np.array([int(x) for x in currentPos])
    aggregate()
    regularize(df)

    return currentPos


if __name__ == "__main__":
    from custom_eval.alleval import all_eval

    all_eval(getMyPosition)
