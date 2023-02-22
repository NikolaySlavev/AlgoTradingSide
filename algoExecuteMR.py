from binance.client import Client
import pandas as pd
import math

from globals import *
from config import *
from algoExecute import *
import strategies


# if we get the data at 1min past 00, then don't we get the same price twice
# how should I do it for exponential data?
# if I want it quicker I need to store the data into local database and the amout of funds I have available?

# usdtNewQuant = client.get_asset_balance(asset = 'USDT')["free"]
# btcNewQuant = client.get_asset_balance(asset = 'BTC')["free"]

if __name__ == "__main__":
    proxies = { 'https' : "http://" + str(username) + ":" + str(password) + "@" + str(proxy) + ":" + str(port)} 
        
    mrOrders = pd.read_csv(mrOrdersPath)
    mrLastOrder = mrOrders.iloc[-1]
    lastSignal = mrLastOrder["signal"]
    usdtQuant = mrLastOrder["usdt"]
    btcQuant = mrLastOrder["btc"]
                
    best_mr_ema_alpha = 0.35720024241140896
    period = math.ceil((2 / best_mr_ema_alpha - 1) * 2)
    symbol = "BTCUSDT"
    
    client = Client(bin_api_key, bin_api_secret, requests_params = {"proxies": proxies})
    df = strategies.BinanceTimeSeries.generate_data(client = client, dataPair = symbol, howLong = period, interval = Client.KLINE_INTERVAL_1HOUR)
    signal = strategies.MeanReversion.MR_exponential_exec(df[PRICE], best_mr_ema_alpha)
            
    if not (signal == BUY and lastSignal != BUY or signal == SELL and lastSignal != SELL):
        order = "Same order as the last one"
    elif signal == BUY:
        order = client.create_order(symbol = symbol, side = "BUY", type = "MARKET", quoteOrderQty = usdtQuant)
        newBtcQuant, newUsdtQuant = getPostOrderQuantities(btcQuant, usdtQuant, order)
        writeOrder(signal, newBtcQuant, newUsdtQuant, order["transactTime"], order["fills"][0]["price"], order["executedQty"], order["fills"][0]["commission"], len(order["fills"]), mrOrdersPath)
    elif signal == SELL:
        order = client.create_order(symbol = symbol, side = "SELL", type = "MARKET",  quantity = btcQuant)
        newBtcQuant, newUsdtQuant = getPostOrderQuantities(btcQuant, usdtQuant, order)            
        writeOrder(signal, newBtcQuant, newUsdtQuant, order["transactTime"], order["fills"][0]["price"], order["executedQty"], order["fills"][0]["commission"], len(order["fills"]), mrOrdersPath)
    elif signal == HOLD:
        order = "HOLD order" 
    else:
        raise Exception("Invalid signal")
    
    print(order)
    writeHistory(signal, order, mrHistoryPath)
    