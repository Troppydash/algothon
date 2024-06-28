import statsmodels.tsa.ardl as ardl

def predict(currentPos, df, ticker, indices, lags, deg):
    if len(df[ticker]) < 200:
        return
    
    if len(indices) > 1:
        matchedSeries = df[indices].pct_change().dropna().values[-200:, :]
    else:
        matchedSeries = df[indices].pct_change().dropna().values[-200:]

    result = ardl.ARDL(
        df[ticker].pct_change().dropna().values[-200:],
        0,
        matchedSeries,
        lags,
        causal=True
    ).fit()

    predict_mu = result.forecast()
    print(predict_mu)
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