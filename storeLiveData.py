import pandas as pd
import sqlalchemy
import asyncio
from binance.client import Client
from binance import BinanceSocketManager
import configparser

def createFrame(msg):
    df = pd.DataFrame([msg])
    df = df.loc[:, ["s", "E", "p"]]
    df.columns = ["symbol", "Time", "Price"]
    df.Price = df.Price.astype(float)
    df.Time = pd.to_datetime(df.Time, unit = "ms")
    return df

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

async def main(socket):
    maxRetry = 5
    retry = 0
    await socket.__aenter__()
    while True:
        msg = await socket.recv()
        if msg['e'] == 'error' and not await processError(socket, retry, maxRetry):
            break
        
        frame = createFrame(msg)
        frame.to_sql("BTCUSDT", engine, if_exists = "append", index = False)
        print(msg)
    
    await socket.__aexit__(None, None, None)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    client = Client(config["BINANCE"]["bin_api_key"], config["BINANCE"]["bin_api_secret"])
    bsm = BinanceSocketManager(client)

    socket = bsm.trade_socket("BTCUSDT")
    engine = sqlalchemy.create_engine("sqlite:///BTCUSDTstream.db")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(socket))
