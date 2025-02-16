import statsmodels.tsa.ardl as ardl

def predict(currentPos, df, ticker, indices, lags, deg, shift=0):
    if len(df[ticker]) < 200 + shift:
        return
    
    matchedSeries = df[indices].pct_change().dropna().values
    # print(matchedSeries.shape)
    if len(indices) > 1:
        end = matchedSeries.shape[0]
        print(end)
        matchedSeries = matchedSeries[(end-200-shift):(end-shift), :]
    else:
        end = len(matchedSeries)
        matchedSeries = matchedSeries[(end-200-shift):(end-shift)]
    # print(matchedSeries)

    result = ardl.ARDL(
        df[ticker].pct_change().dropna().values[-200:],
        0,
        matchedSeries,
        lags,
        causal=True
    ).fit()

    predict_mu = result.forecast()
    est = predict_mu
    val = df[ticker].values[-1]
    vol = int(10000 / val)
    trans = 0.001 * deg * val * vol

    pforecast = est * val
    print("Predict", pforecast)
    print("Previous actual", df[ticker].values[-1] - df[ticker].values[-2])
    if abs(vol * pforecast) > trans:
        if pforecast > 0:
            currentPos[ticker] = vol
        else:
            currentPos[ticker] = -vol