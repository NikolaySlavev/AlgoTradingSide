import pandas as pd
from binance.client import Client
from datetime import datetime, timezone

from globals import *
from config import bin_api_key, bin_api_secret, proxy, port, username, password, tfOrdersPath, tfHistoryPath
from algoExecute import *
import strategies


if __name__ == "__main__":
    proxies = { 'https' : "http://" + str(username) + ":" + str(password) + "@" + str(proxy) + ":" + str(port)} 
    
    tfOrders = pd.read_csv(tfOrdersPath)
    tfLastOrder = tfOrders.iloc[-1]
    lastSignal = tfLastOrder["signal"]
    usdtQuant = tfLastOrder["usdt"]
    btcQuant = tfLastOrder["btc"]
                    
    best_tf_sma_period = 570
    best_tf_bb_period = 511
    best_tf_bb_std = 0.0093612
    best_tf_rsi_period = 5
    
    client = Client(bin_api_key, bin_api_secret, requests_params = {"proxies": proxies})
    df = strategies.BinanceTimeSeries.generate_data(client = client, dataPair = "BTCUSDT", howLong = 24, interval = Client.KLINE_INTERVAL_1HOUR)
    signal = strategies.TrendFollowing.TF_simple_exec(df[PRICE], best_tf_sma_period)
    #tf_bb_rsi_signal = strategies.TrendFollowing.TF_bb_rsi_exec(df[PRICE], best_tf_bb_period, best_tf_bb_std, best_tf_rsi_period)
    
    if not (signal == BUY and lastSignal != BUY or signal == SELL and lastSignal != SELL):
        order = "Same order as the last one"
    elif signal == BUY:
        order = client.create_order(symbol = "BTCUSDT", side = "BUY", type = "MARKET", quoteOrderQty = usdtQuant)
        newBtcQuant, newUsdtQuant = getPostOrderQuantities(btcQuant, usdtQuant, order)
        writeOrder(signal, newBtcQuant, newUsdtQuant, order["transactTime"], order["fills"][0]["price"], order["executedQty"], order["fills"][0]["commission"], len(order["fills"]), tfOrdersPath)
    elif signal == SELL:
        order = client.create_order(symbol = "BTCUSDT", side = "SELL", type = "MARKET",  quantity = btcQuant)
        newBtcQuant, newUsdtQuant = getPostOrderQuantities(btcQuant, usdtQuant, order)
        writeOrder(signal, newBtcQuant, newUsdtQuant, order["transactTime"], order["fills"][0]["price"], order["executedQty"], order["fills"][0]["commission"], len(order["fills"]), tfOrdersPath)
    elif signal == HOLD:
        order = "HOLD order"
    else:
        raise Exception("Invalid signal")

    print(order)
    writeHistory(signal, order, tfHistoryPath)
