from collections import defaultdict

import numpy as np
import pandas as pd

limit = 10000
SCALER = 5
INF = 1000000000
commRate = 0.0010

def getUnitTradeVolume(prices, share1, share2, trade1, trade2):
    global limit

    price1 = prices[share2][-1]
    price2 = prices[share1][-1]

    # Get available volume
    vol1 = limit/ price1
    vol2 = limit/ price2

    unitTrade = min(vol1/trade1, vol2/trade2)
    return unitTrade * SCALER


better_pair__last = defaultdict(lambda: 0)
better_pair__amount = defaultdict(lambda: (None, 0))
def better_pair(prices, share1, share2, beta, lower, middle, upper, trade1, trade2):
    global better_pair__last, better_pair__amount
    
    delta = prices[share1][-1] - beta * prices[share2][-1]
    unitTrade = getUnitTradeVolume(prices, share1, share2, trade1, trade2)
    volume1 = unitTrade * trade1
    volume2 = unitTrade * trade2

    if delta < lower and better_pair__last[share1, share2] != -1:
        better_pair__last[share1, share2] = -1
        better_pair__amount[f"{share1}-{share2}-1"] = (share1, volume1)
        better_pair__amount[f"{share1}-{share2}-2"] = (share2, -volume2)
    
    if better_pair__last[share1, share2] < 0 and delta >= middle or better_pair__last[share1, share2] > 0 and delta <= middle:
        better_pair__last[share1, share2] = 0
        better_pair__amount[f"{share1}-{share2}-1"] = (share1, int(better_pair__amount[f"{share1}-{share2}-1"][1] // 3))
        better_pair__amount[f"{share1}-{share2}-2"] = (share2, int(better_pair__amount[f"{share1}-{share2}-2"][1] // 3))

    if delta > upper and better_pair__last[share1, share2] != 1:
        better_pair__last[share1, share2] = 1
        better_pair__amount[f"{share1}-{share2}-1"] = (share1, -volume1)
        better_pair__amount[f"{share1}-{share2}-2"] = (share2, volume2)

def better_pair_aggregate():
    currentPos = np.zeros(50)
    for key, value in better_pair__amount.items():
        currentPos[value[0]] += value[1]
    return currentPos


