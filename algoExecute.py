import csv
from datetime import datetime, timezone


def writeOrder(signal, newBtcQuant, newUsdtQuant, time, price, executedQty, commision, fillsLength, mrOrdersPath):
    with open(mrOrdersPath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([signal, newBtcQuant, newUsdtQuant, time, price, executedQty, commision, fillsLength])

def writeHistory(signal, order, historyPath):
    with open(historyPath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([signal, datetime.now(timezone.utc), order])

def getPostOrderQuantities(btcQuant, usdtQuant, order):
    if order["side"] == "SELL":
        newBtcQuant = btcQuant - float(order["executedQty"])
        newUsdtQuant = usdtQuant + float(order["cummulativeQuoteQty"])
    elif order["side"] == "BUY":
        newBtcQuant = btcQuant + float(order["executedQty"])
        newUsdtQuant = usdtQuant - float(order["cummulativeQuoteQty"])
    else:
        raise Exception("Invalid order side", order["side"])
    
    if order["origQty"] != order["executedQty"]:
        raise Exception("Executed quantities are not equal", order["origQty"], order["executedQty"], str(order))
    
    return newBtcQuant, newUsdtQuant
