from util.constants import LIMIT

def setVolume(newVolume, price):
    if newVolume > 0:
        return int(min(LIMIT//price, newVolume))
    else:
        return int(max(-LIMIT//price, newVolume))