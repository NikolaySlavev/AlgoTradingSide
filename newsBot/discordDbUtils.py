#import csv
import pandas as pd
import sshtunnel
import MySQLdb


BASESTEPSIZE = 0.00001
MAXERRORLENGTH = 2000

sshtunnel.SSH_TIMEOUT = 5.0
sshtunnel.TUNNEL_TIMEOUT = 5.0

class DiscordNewsInfo():
    def __init__(self, channelName, channelId, lastNewsDate):
        self.channelName = channelName
        self.channelId = channelId
        self.lastNewsDate = lastNewsDate

class DiscordNewsLog():
    def __init__(self, channelName, channelId, newsDate, newsTraded, newsContent, newsResponse, newsQuestions):
        self.channelName = channelName
        self.channelId = channelId
        self.newsDate = newsDate
        self.newsTraded = newsTraded
        self.newsContent = newsContent
        self.newsResponse = newsResponse
        self.newsQuestions = newsQuestions


def updateDiscordNewsInfo(channelId, lastNewsDate, config):
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
            sql = "UPDATE discord_news_info SET last_news_date = (%s) WHERE channel_id = (%s)"
            # sql = "INSERT INTO discord_news_info \
            #     (channel_name, channel_id, last_news_date) VALUES \
            #     (%s, %s, %s)"
            
            #val = [discordNewsInfo.channelName, discordNewsInfo.channelId, discordNewsInfo.lastNewsDate]
            val = [lastNewsDate, channelId]
            mycursor.execute(sql, val)
            mydb.commit()
        finally:
            mydb.close()
            

def writeDiscordNewsLog(discordNewsLog, config):
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
            
            val = [discordNewsLog.strategyName, discordNewsLog.asofdate, discordNewsLog.tradeSignal, discordNewsLog.pairName, discordNewsLog.log, discordNewsLog.error]
            mycursor.execute(sql, val)
            mydb.commit()
        finally:
            mydb.close()

def getDiscordNewsInfo(config):
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
            queryResult = pd.read_sql(f"select * from discord_news_info", mydb)
        finally:
            mydb.close()
        
    return queryResult