from imports import *
from execute.algoExecute import *
from strategies.MeanReversion import MeanReversion
from timeSeries.BinanceTimeSeries import BinanceTimeSeries


# if we get the data at 1min past 00, then don't we get the same price twice
# how should I do it for exponential data?
# if I want it quicker I need to store the data into local database and the amout of funds I have available?

def main(strategyName, bestMrParam):
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    try:
        proxies = { 'https' : "http://" + str(username) + ":" + str(password) + "@" + str(proxy) + ":" + str(port)}
        lastOrderInfo = getLastTransaction(config, strategyName)
        
        period = math.ceil((2 / bestMrParam - 1) * 2)
        
        btcRem = lastOrderInfo.longInventory % BASESTEPSIZE
        lastOrderInfo.longInventory -= btcRem
        
        untilThisDate = datetime.datetime.now()
        sinceThisDate = untilThisDate - datetime.timedelta(days = period)
        
        client = Client(bin_api_key, bin_api_secret, requests_params = {"proxies": proxies})
        info = client.get_symbol_info(lastOrderInfo.pairName)
        
        df = BinanceTimeSeries.generateData(client = client, config = config, dataPair = lastOrderInfo.pairName, sinceThisDate = sinceThisDate, untilThisDate = untilThisDate, interval = Client.KLINE_INTERVAL_1HOUR)
        signal = MeanReversion.MR_exponential_exec(df[PRICE], bestMrParam)
                
        if not (signal == BUY and lastOrderInfo.tradeSignal != BUY or signal == SELL and lastOrderInfo.tradeSignal != SELL):
            order = SAMEORDERSTR
        elif signal == BUY:
            # need to round because it gives a precision error (don't overwrite the real value)
            order = client.create_order(symbol = lastOrderInfo.pairName, side = "BUY", type = "MARKET", quoteOrderQty = round(lastOrderInfo.shortInventory, info["quoteAssetPrecision"]))
        elif signal == SELL:
            order = client.create_order(symbol = lastOrderInfo.pairName, side = "SELL", type = "MARKET",  quantity = round(lastOrderInfo.longInventory, info["baseAssetPrecision"]))
        else:
            raise Exception("Invalid signal")

        print(order)
        if order != SAMEORDERSTR:
            newBtcQuant, newUsdtQuant = getPostOrderQuantities(lastOrderInfo.longInventory + btcRem, lastOrderInfo.shortInventory, order)
            postOrderInfo = OrderInfo(strategyName = strategyName, 
                                            asofdate = datetime.datetime.fromtimestamp(order["transactTime"] / 1000),
                                            tradeSignal = signal, 
                                            longInventory = newBtcQuant, 
                                            shortInventory = newUsdtQuant, 
                                            price = float(order["fills"][0]["price"]),
                                            amount = float(order["executedQty"]),
                                            commission = float(order["fills"][0]["commission"]),
                                            pairName = lastOrderInfo.pairName,
                                            fillsNumber = len(order["fills"]))
                
            writeOrder(postOrderInfo, config)
        
        postHistoryInfo = HistoryInfo(strategyName = strategyName, asofdate = datetime.datetime.now(), tradeSignal = signal, pairName = lastOrderInfo.pairName, log = order, error = 0)
        
    except Exception as ex:
        exStr = str(ex)[:MAXERRORLENGTH] if len(str(ex)) > MAXERRORLENGTH else str(ex)
        postHistoryInfo = HistoryInfo(strategyName = strategyName, asofdate = datetime.datetime.now(), tradeSignal = 0, pairName = "", log = exStr, error = 1)
    
    writeHistory(postHistoryInfo, config)
    
if __name__ == "__main__":
    strategyName = "MR_EMA_0.3572_HOUR_15_USDT"
    best_mr_ema_alpha = 0.35720024241140896
    main(strategyName, best_mr_ema_alpha)
