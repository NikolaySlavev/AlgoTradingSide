from imports import *
from execute.algoExecute import *
from strategies.MeanReversion import MeanReversion
from timeSeries.BinanceTimeSeries import BinanceTimeSeries


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
    
    baseStepSize = 0.00001
    btcRem = btcQuant % baseStepSize
    btcQuant -= btcRem
    
    untilThisDate = datetime.datetime.now()
    sinceThisDate = untilThisDate - datetime.timedelta(days = period)
    
    client = Client(bin_api_key, bin_api_secret, requests_params = {"proxies": proxies})
    info = client.get_symbol_info(symbol)
    
    df = BinanceTimeSeries.generateData(client = client, dataPair = symbol, sinceThisDate = sinceThisDate, untilThisDate = untilThisDate, interval = Client.KLINE_INTERVAL_1HOUR)
    signal = MeanReversion.MR_exponential_exec(df[PRICE], best_mr_ema_alpha)
            
    if not (signal == BUY and lastSignal != BUY or signal == SELL and lastSignal != SELL):
        order = "Same order as the last one"
    elif signal == BUY:
        # need to round because it gives a precision error (don't overwrite the real value)
        order = client.create_order(symbol = symbol, side = "BUY", type = "MARKET", quoteOrderQty = round(usdtQuant, info["quoteAssetPrecision"]))
        newBtcQuant, newUsdtQuant = getPostOrderQuantities(btcQuant + btcRem, usdtQuant, order)
        writeOrder(signal, newBtcQuant, newUsdtQuant, order["transactTime"], order["fills"][0]["price"], order["executedQty"], order["fills"][0]["commission"], len(order["fills"]), mrOrdersPath)
    elif signal == SELL:
        order = client.create_order(symbol = symbol, side = "SELL", type = "MARKET",  quantity = round(btcQuant, info["baseAssetPrecision"]))
        newBtcQuant, newUsdtQuant = getPostOrderQuantities(btcQuant + btcRem, usdtQuant, order)            
        writeOrder(signal, newBtcQuant, newUsdtQuant, order["transactTime"], order["fills"][0]["price"], order["executedQty"], order["fills"][0]["commission"], len(order["fills"]), mrOrdersPath)
    elif signal == HOLD:
        order = "HOLD order" 
    else:
        raise Exception("Invalid signal")
    
    print(order)
    writeHistory(signal, order, mrHistoryPath)
    
    
    
# 'symbol':
# 'BTCUSDT'
# 'orderId':
# 21791610938
# 'orderListId':
# -1
# 'clientOrderId':
# 'o6G3XdbncUwVYQZiAQuT4V'
# 'transactTime':
# 1689463666890
# 'price':
# '0.00000000'
# 'origQty':
# '0.00050000'
# 'executedQty':
# '0.00050000'
# 'cummulativeQuoteQty':
# '15.14496500'
# 'status':
# 'FILLED'
# 'timeInForce':
# 'GTC'
# 'type':
# 'MARKET'
# 'side':
# 'SELL'
# 'workingTime':
# 1689463666890
# 'fills':
# [{'price': '30289.93000000', 'qty': '0.00050000', 'commission': '0.01514497', 'commissionAsset': 'USDT', 'tradeId': 3172699611}]
# 'selfTradePreventionMode':
# 'NONE'