#import csv
import datetime
import mysql.connector
import pandas as pd
import sshtunnel
import MySQLdb


BASESTEPSIZE = 0.00001
MAXERRORLENGTH = 2000

sshtunnel.SSH_TIMEOUT = 5.0
sshtunnel.TUNNEL_TIMEOUT = 5.0


class OrderInfo():
    #TODO: Currently only accepts the first price and commission of the executed fills. We don't record subsequent fills
    def __init__(self, strategyName, asofdate, tradeSignal, pairName, longInventory, shortInventory, price, amount, commission, fillsNumber):
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
          
    @classmethod
    def fromDb(cls, queryResult):
        return cls(strategyName = queryResult["strategy_unique_name"], 
                   asofdate = queryResult["asofdate"], 
                   tradeSignal = queryResult["trade_signal"],
                   longInventory = queryResult["long_inventory"],
                   shortInventory = queryResult["short_inventory"],
                   price = queryResult["price"],
                   amount = queryResult["amount"],
                   commission = queryResult["commission"],
                   pairName = queryResult["pair_name"],
                   fillsNumber = queryResult["fills_number"])


class HistoryInfo():
    def __init__(self, strategyName, asofdate, tradeSignal, pairName, log, error):
        self.strategyName = strategyName
        self.asofdate = asofdate
        self.tradeSignal = tradeSignal        
        self.pairName = pairName
        self.log = str(log)
        self.error = error
        

def writeOrder(postOrderInfo, config):
    #mydb = mysql.connector.connect(user = config["MYSQL"]["database_user"], password = config["MYSQL"]["database_password"], host = config["MYSQL"]["database_host"], database = config["MYSQL"]["database_name"])
    with sshtunnel.SSHTunnelForwarder(
    (config["MYSQL"]["ssh_host"]), 
    ssh_username=config["MYSQL"]["ssh_user"], ssh_password=config["MYSQL"]["ssh_password"], 
    remote_bind_address=(config["MYSQL"]["database_host"], 3306)) as tunnel:
        mydb = MySQLdb.connect(
            user=config["MYSQL"]["database_user"],
            passwd=config["MYSQL"]["database_password"],
            host='127.0.0.1', port=tunnel.local_bind_port,
            db=config["MYSQL"]["database_name"])
        try:
            mycursor = mydb.cursor()
            sql = "INSERT INTO transactions_binance_btc_usdt \
                (strategy_unique_name, asofdate, trade_signal, long_inventory, short_inventory, price, amount, commission, pair_name, fills_number) VALUES \
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                
            # DONE add the db remotely and try to connect to it locally and write to it remotely
            # fix mr backtesting
            # populate the binance data in the db
            # add a script to populate it daily? and we only fetch from the current date
            # problem that python anywhere scheduler is hourly. Can it run all the time and have internal scheduler?
            # compare the speed improvement
            val = [postOrderInfo.strategyName, 
                postOrderInfo.asofdate, 
                postOrderInfo.tradeSignal, 
                postOrderInfo.longInventory, 
                postOrderInfo.shortInventory, 
                postOrderInfo.price,
                postOrderInfo.amount,
                postOrderInfo.commission,
                postOrderInfo.pairName,
                postOrderInfo.fillsNumber]
            mycursor.execute(sql, val)
            mydb.commit()
        finally:
            mydb.close()

def writeHistory(postHistoryInfo, config):
    #mydb = mysql.connector.connect(user = config["MYSQL"]["database_user"], password = config["MYSQL"]["database_password"], host = config["MYSQL"]["database_host"], database = config["MYSQL"]["database_name"])
    with sshtunnel.SSHTunnelForwarder(
    (config["MYSQL"]["ssh_host"]), 
    ssh_username=config["MYSQL"]["ssh_user"], ssh_password=config["MYSQL"]["ssh_password"], 
    remote_bind_address=(config["MYSQL"]["database_host"], 3306)) as tunnel:
        mydb = MySQLdb.connect(
            user=config["MYSQL"]["database_user"],
            passwd=config["MYSQL"]["database_password"],
            host='127.0.0.1', port=tunnel.local_bind_port,
            db=config["MYSQL"]["database_name"])
        try:
            mycursor = mydb.cursor()
            sql = "INSERT INTO logs_binance_btc_usdt \
                (strategy_unique_name, asofdate, trade_signal, pair_name, log, error) VALUES \
                (%s, %s, %s, %s, %s, %s)"
            
            val = [postHistoryInfo.strategyName, postHistoryInfo.asofdate, postHistoryInfo.tradeSignal, postHistoryInfo.pairName, postHistoryInfo.log, postHistoryInfo.error]
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

def getLastTransaction(config, strategyName):
    #mydb = mysql.connector.connect(user = config["MYSQL"]["database_user"], password = config["MYSQL"]["database_password"], host = config["MYSQL"]["database_host"], database = config["MYSQL"]["database_name"])
    with sshtunnel.SSHTunnelForwarder(
    (config["MYSQL"]["ssh_host"]), 
    ssh_username=config["MYSQL"]["ssh_user"], ssh_password=config["MYSQL"]["ssh_password"], 
    remote_bind_address=(config["MYSQL"]["database_host"], 3306)) as tunnel:
        mydb = MySQLdb.connect(
            user=config["MYSQL"]["database_user"],
            passwd=config["MYSQL"]["database_password"],
            host='127.0.0.1', port=tunnel.local_bind_port,
            db=config["MYSQL"]["database_name"])
        try:
            queryResult = pd.read_sql(f"select * from transactions_binance_btc_usdt where strategy_unique_name = '{strategyName}' and asofdate = (select max(asofdate) from transactions_binance_btc_usdt)", mydb)
        finally:
            mydb.close()
        
    if len(queryResult) != 1:
        raise Exception(f"Cannot find appropriate last transaction for {strategyName}. The selected length was " + str(len(queryResult)))
    
    orderInfo = OrderInfo.fromDb(queryResult.iloc[0])
    return orderInfo
