from imports import *
from execute.algoExecute import *
from strategies.TrendFollowing import TrendFollowing
from timeSeries.BinanceTimeSeries import BinanceTimeSeries

class PostOrderInfo():
    #TODO: Currently only accepts the first price and commission of the executed fills. We don't record subsequent fills
    def __init__(self, strategyName, asofdate, tradeSignal, longInventory, shortInventory, price, amount, commission, pairName, fillsNumber):
        self.strategyName = strategyName
        self.asofdate = asofdate
        self.tradeSignal = tradeSignal
        self.longInventory = longInventory
        self.shortInventory = shortInventory
        self.price = price
        self.amount = amount
        self.commission = commission
        self.pairName = pairName
        self.fillsNumber = fillsNumber        
        
        


if __name__ == "__main__":
    proxies = { 'https' : "http://" + str(username) + ":" + str(password) + "@" + str(proxy) + ":" + str(port)} 
    
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    strategyName = "TF_SMA_570_HOUR_15_USDT"

    tfLastOrder = None
    
    # IN CONFIG DATABASE HAVE DIFFERENT DATABASES FOR LOGS AND TRANSACTIONS AND DATA
    mydb = mysql.connector.connect(user = config["root"], password = config["password"], host = config["host"], database = config["database"])
    try:
        mycursor = mydb.cursor()
        # DO SELECT STATEMENT        
        sql = f"select * from transactions_binance_btc_usdt where strategy_unique_name = {strategyName} and asofdate = (select max(asofdate) from transactions_binance_btc_usdt)" 
        mycursor.execute(sql)
        myresult = mycursor.fetchall()
        for x in myresult:
            tfLastOrder = x
    finally:
        mydb.close()
    
    #tfOrders = pd.read_csv(tfOrdersPath)
    #tfLastOrder = tfOrders.iloc[-1]
    lastSignal = tfLastOrder["signal"]
    usdtQuant = tfLastOrder["usdt"]
    btcQuant = tfLastOrder["btc"]
                    
    best_tf_sma_period = 570
    best_tf_bb_period = 511
    best_tf_bb_std = 0.0093612
    best_tf_rsi_period = 5
    symbol = "BTCUSDT"
    
    baseStepSize = 0.00001
    btcRem = btcQuant % baseStepSize
    btcQuant -= btcRem
    
    untilThisDate = datetime.datetime.now()
    sinceThisDate = untilThisDate - datetime.timedelta(days = 24)
    
    client = Client(bin_api_key, bin_api_secret, requests_params = {"proxies": proxies})
    info = client.get_symbol_info(symbol)
    
    df = BinanceTimeSeries.generateData(client = client, dataPair = symbol, sinceThisDate = sinceThisDate, untilThisDate = untilThisDate, interval = Client.KLINE_INTERVAL_1HOUR)
    signal = TrendFollowing.TF_simple_exec(df[PRICE], best_tf_sma_period)
    #tf_bb_rsi_signal = strategies.TrendFollowing.TF_bb_rsi_exec(df[PRICE], best_tf_bb_period, best_tf_bb_std, best_tf_rsi_period)
    
    if not (signal == BUY and lastSignal != BUY or signal == SELL and lastSignal != SELL):
        order = "Same order as the last one"
    elif signal == BUY:
        # need to round because it gives a precision error (don't overwrite the real value)
        order = client.create_order(symbol = "BTCUSDT", side = "BUY", type = "MARKET", quoteOrderQty = round(usdtQuant, info["quoteAssetPrecision"]))
    elif signal == SELL:
        order = client.create_order(symbol = "BTCUSDT", side = "SELL", type = "MARKET",  quantity = round(btcQuant, info["baseAssetPrecision"]))
    elif signal == HOLD:
        order = "HOLD order"
    else:
        raise Exception("Invalid signal")

    if signal != HOLD:
        newBtcQuant, newUsdtQuant = getPostOrderQuantities(btcQuant + btcRem, usdtQuant, order)
        postOrderInfo = PostOrderInfo(strategyName = strategyName, 
                                        asofdate = order["transactTime"], 
                                        tradeSignal = signal, 
                                        longInventory = newBtcQuant, 
                                        shortInventory = newUsdtQuant, 
                                        price = order["fills"][0]["price"],
                                        amount = order["executedQty"],
                                        commission = order["fills"][0]["commission"],
                                        parName = symbol,
                                        fillsNumber = len(order["fills"]))
              
        writeOrder(postOrderInfo, config)

    print(order)
    writeHistory(signal, order, tfHistoryPath)
