import pandas as pd
import datetime
import sqlalchemy
from binance.client import Client
import configparser


def getHistoricalData(client, dataPair, howLong, interval):
    # Calculate the timestamps for the binance api function
    untilThisDate = datetime.datetime.now()
    sinceThisDate = untilThisDate - datetime.timedelta(days = howLong)
    
    # Execute the query from binance - timestamps must be converted to strings
    candle = client.get_historical_klines(dataPair, interval, str(sinceThisDate), str(untilThisDate))

    # Create a dataframe to label all the columns returned by binance so we work with them later.
    df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])
    # as timestamp is returned in ms, let us convert this back to proper timestamps.
    dateTimeFormat = "%d/%m/%y"
    df.dateTime = pd.to_datetime(df.dateTime, unit='ms').dt.strftime(dateTimeFormat)
    df.set_index('dateTime', inplace = True)
    df.open = df.open.astype(float)
    df.close = df.close.astype(float)
    df.high = df.high.astype(float)
    df.low = df.low.astype(float)

    # Get rid of columns we do not need
    df = df.drop(['closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
    return df    


if __name__ == "__main__":
    dataPair = "BTCUSDT"
    
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    client = Client(config["BINANCE"]["bin_api_key"], config["BINANCE"]["bin_api_secret"])
    engine = sqlalchemy.create_engine("sqlite:///" + dataPair + "stream.db")

    df = getHistoricalData(client, dataPair, 30)
    df.to_sql("BTCUSDT", engine, if_exists = "replace", index = False)
    print(df)
    