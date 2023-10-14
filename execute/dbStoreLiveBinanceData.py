import datetime, traceback, asyncio, configparser
from binance.client import Client
from binance import BinanceSocketManager
from execute.algoExecute import *


async def processError(socket, retry):
    maxRetry = 5
    isBreak = False
    retry += 1
    await socket.__aexit__(None, None, None)
    if retry == maxRetry:
        isBreak = True
    else:
        await socket.__aenter__()
            
    return isBreak

async def main(config):
    maxRetry = 5
    retry = 0
    prevPrice = 0
    prevDt = 0
    
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
            
            client = Client(config["BINANCE"]["bin_api_key"], config["BINANCE"]["bin_api_secret"])
            bsm = BinanceSocketManager(client)
            socket = bsm.trade_socket("BTCUSDT")
            await socket.__aenter__()
            while True:
                msg = await socket.recv()
                if msg['e'] == 'error' and not await processError(socket, retry, maxRetry):
                    raise Exception(f"Cannot fetch prices. Error: {msg}")
                
                dt = datetime.datetime.fromtimestamp(msg['E'] / 1000).replace(microsecond = 0)
                price = float(msg["p"])
                if price == prevPrice or dt == prevDt:
                    continue
                
                sql = "INSERT INTO data_realtime_binance_btc_usdt (`time`, `price`) VALUES (%s, %s)"
                val = [dt, price]
                mycursor.execute(sql, val)
                mydb.commit()
                
                prevPrice = price
                prevDt = dt
        except Exception as ex:
            print(ex)
            exStr = f"SHORT: {ex} \nFULL: {traceback.format_exc()}"
            exStr = str(exStr)[:MAXERRORLENGTH] if len(str(exStr)) > MAXERRORLENGTH else str(exStr)
            postHistoryInfo = HistoryInfo(strategyName = "REALTIME DATA", asofdate = datetime.datetime.now(), tradeSignal = 0, pairName = "BTCUSDT", log = exStr, error = 1)
            writeHistory(postHistoryInfo, config)
        finally:
            mydb.close()
    

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(config))
