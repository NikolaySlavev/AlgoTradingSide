#import csv
import datetime
import mysql.connector

def writeOrder(postOrderInfo, config):
    mydb = mysql.connector.connect(user = config["root"], password = config["password"], host = config["host"], database = config["database"])    
    try:
        mycursor = mydb.cursor()
        sql = "INSERT INTO transactions_binance_btc_usdt \
            (strategy_unique_name, asofdate, trade_signal, long_inventory, short_inventory, price, amount, commission, pair_name, fills_number) VALUES \
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            
        # FINISH THIS AND writeHistory. Then populate some entry in db to test the select. Then test the write.
        # test it in real transaction
        # make it work for mr too
        # add the db remotelyand try to connect to it locally and write to it remotely
        # fix mr backtesting
        # populate the binance data in the db
        # add a script to populate it daily? and we only fetch from the current date
        # problem that python anywhere scheduler is hourly. Can it run all the time and have internal scheduler?
        # compare the speed improvement
        val = [postOrderInfo.strategyName, postOrderInfo.asofdate, postOrderInfo.tradeSignal, str(100), str(2), str(12), str(1), str(3), "TEST", str(1)]
        mycursor.execute(sql, val)
        mydb.commit()
    finally:
        mydb.close()

def writeHistory(signal, order, historyPath, config):
    mydb = mysql.connector.connect(user = config["root"], password = config["password"], host = config["host"], database = config["database"])    
    try:
        mycursor = mydb.cursor()
        sql = "INSERT INTO transactions_binance_btc_usdt \
            (strategy_unique_name, asofdate, trade_signal, long_inventory, short_inventory, price, amount, commission, pair_name, fills_number) VALUES \
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        val = ["test", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "BUY", str(100), str(2), str(12), str(1), str(3), "TEST", str(1)]
        mycursor.execute(sql, val)
        mydb.commit()
    finally:
        mydb.close()
    
def getCommission(fills):
    commission = 0
    for fill in fills:
        commission += float(fill["commission"])
    
    return commission

def getPostOrderQuantities(btcQuant, usdtQuant, order):
    commission = getCommission(order["fills"])
    
    if order["side"] == "SELL":
        newBtcQuant = btcQuant - float(order["executedQty"])
        newUsdtQuant = usdtQuant + float(order["cummulativeQuoteQty"]) - commission
    elif order["side"] == "BUY":
        newBtcQuant = btcQuant + float(order["executedQty"]) - commission
        newUsdtQuant = usdtQuant - float(order["cummulativeQuoteQty"])
    else:
        raise Exception("Invalid order side", order["side"])
    
    if order["origQty"] != order["executedQty"]:
        raise Exception("Executed quantities are not equal", order["origQty"], order["executedQty"], str(order))
    
    return newBtcQuant, newUsdtQuant
