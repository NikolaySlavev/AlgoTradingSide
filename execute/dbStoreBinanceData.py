import configparser
from binance.client import Client
import sshtunnel
import MySQLdb
from sqlalchemy import create_engine
from datetime import datetime
import traceback

from globals import *
from timeSeries.BinanceTimeSeries import BinanceTimeSeries
from execute.algoExecute import *


BASESTEPSIZE = 0.00001
MAXERRORLENGTH = 2000

sshtunnel.SSH_TIMEOUT = 5.0
sshtunnel.TUNNEL_TIMEOUT = 5.0


def main(interval, pairName, strategyName):
    print("STARTING dbStoreBinanceData " + str(interval))
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    try:
        client = Client(config["BINANCE"]["bin_api_key"], config["BINANCE"]["bin_api_secret"])
        timeSeries = BinanceTimeSeries.fromHowLong(client = client, config = config, dataPair = pairName, howLong = 600, interval = interval, numSplits = 0)
        dfAll = timeSeries.df.reset_index()
        dfAll = dfAll[['dateTime', 'closeTime', 'open', 'close', 'high', 'low', 'volume']]
        dfAll = dfAll.rename(columns={"dateTime": "open_time", "closeTime": "close_time", "open": "open_price", "close": "close_price", "high": "high_price", "low": "low_price", "volume": "volume"})
        dfAll["interval"] = interval
        
        # IMPORTANT: binance returns the last candle that hasn't been closed yet (e.g. time is 11:30 and we want the candle from 11 to 12, but it returns the candle from 11 to 11:30)
        dfAll = dfAll[:-1]
        with sshtunnel.SSHTunnelForwarder(
            (config["MYSQL"]["ssh_host"]), 
            ssh_username=config["MYSQL"]["ssh_user"], ssh_password=config["MYSQL"]["ssh_password"], 
            remote_bind_address=(config["MYSQL"]["database_host"], 3306)) as tunnel:
            
                engine = create_engine(f'mysql+mysqldb://{config["MYSQL"]["database_user"]}:{config["MYSQL"]["database_password"]}@127.0.0.1:{tunnel.local_bind_port}/{config["MYSQL"]["database_name"]}')
                with engine.begin() as conn:
                    conn.execute("truncate tmp_data_binance_btc_usdt")
                
                dfAll.to_sql("tmp_data_binance_btc_usdt", con = engine, if_exists = "append", index = False)
                with engine.begin() as conn:
                    conn.execute("call etl_data_binance_btc_usdt")

        log = f"length: {len(dfAll)}, interval: {interval}, start_time: {dfAll['open_time'].head(1)}, end_time: {dfAll['close_time'].tail(1)}"
        postHistoryInfo = HistoryInfo(strategyName = strategyName, asofdate = datetime.datetime.now(), tradeSignal = 0, pairName = pairName, log = log, error = 0)
                
    except Exception as ex:
        exStr = f"SHORT: {ex} \nFULL: {traceback.format_exc()}"
        exStr = str(exStr)[:MAXERRORLENGTH] if len(str(exStr)) > MAXERRORLENGTH else str(exStr)
        postHistoryInfo = HistoryInfo(strategyName = strategyName, asofdate = datetime.datetime.now(), tradeSignal = 0, pairName = pairName, log = exStr, error = 1)
        
    writeHistory(postHistoryInfo, config)

if __name__ == "__main__":
    pairName = "BTCUSDT"
    for interval in [Client.KLINE_INTERVAL_1MINUTE, Client.KLINE_INTERVAL_1HOUR]:
        strategyName = f"dbStoreBinanceData_{pairName}_{interval}"
        main(interval, pairName, strategyName)
