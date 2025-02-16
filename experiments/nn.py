import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class StockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(50 * 3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.stack(x)

class StockNetwork27(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(50 * 3, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.stack(x)


model = StockNetwork()
state = torch.load('da/model.pth')
model.load_state_dict(state)


currentPos = np.zeros(50)

def getMyPosition(prices):
    global currentPos

    nins, nt = prices.shape

    df = pd.DataFrame(prices.T, columns=np.arange(50))

    # trade 18
    # get last 10
    last = df.pct_change().iloc[-3:, :] * 100
    X = torch.from_numpy(last.values.T.flatten()).type(torch.float)
    # print(X.shape)
    model.eval()
    with torch.no_grad():
        pred = model(X)
        pct = pred.item() / 100

        ticker = 38
        deg = 0.9

        val = df[ticker].values[-1]
        vol = int(10000 / val)
        trans = 0.001 * deg * val * vol

        pforecast = pct * val
        if abs(vol * pforecast) > trans:
            if pforecast > 0:
                currentPos[ticker] = vol
            else:
                currentPos[ticker] = -vol


    return currentPos