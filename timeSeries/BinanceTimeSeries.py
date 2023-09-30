from imports import *
from timeSeries.TimeSeries import TimeSeries

import sshtunnel
import MySQLdb


renameDbData = {"open_time": "dateTime", "close_time": "closeTime" , "open_price": "open", "close_price": "close", "high_price": "high", "low_price": "low"}

class BinanceTimeSeries(TimeSeries):
    def __init__(self, client, config, dataPair, sinceThisDate, untilThisDate, interval, numSplits, onlyBinance = False):
        super(BinanceTimeSeries, self).__init__()
        
        if onlyBinance:
            self.df = BinanceTimeSeries.generateData(client, dataPair, sinceThisDate, untilThisDate, interval)
        else:
            self.dfDb = BinanceTimeSeries.generateDbData(config, sinceThisDate, untilThisDate, interval)
            self.dfDb.rename(columns = renameDbData, inplace = True)
            self.dfDb.drop("interval", axis = 1, inplace = True)
            self.dfDb["price"] = self.dfDb["close"]
            self.dfDb.set_index('dateTime', inplace = True)
            self.dfDb.sort_index(inplace = True)
            lastPresentCloseDate = self.dfDb["closeTime"].iloc[-1]
            self.dfBinance = BinanceTimeSeries.generateData(client, dataPair, lastPresentCloseDate, untilThisDate, interval)
            self.df = pd.concat([self.dfDb,self.dfBinance])
        
        self.df[RETURN] = self.getReturns(useSet = ALL)
        self.dataFullNp = TimeSeries.dfToNp(self.df)
        self.columnsNp = TimeSeries.dfColumnsToNp(self.df)
        
        if numSplits == 0:
            self.dataTrainList.append(self.dataFullNp)
            self.dataTestList.append(self.dataFullNp)
            self.dataTrainTestList.append(self.dataFullNp)
            self.setCurrentTrainTestDataNp(0)
        elif numSplits == 1:
            cut = int(len(self.df) * 0.7)
            self.dataTrainList.append(self.dataFullNp[:cut])
            self.dataTestList.append(self.dataFullNp[cut:])
            self.dataTrainTestList.append(np.concatenate([self.dataFullNp[:cut], self.dataFullNp[cut:]]))
            self.setCurrentTrainTestDataNp(0)
        else:
            self.tss = TimeSeriesSplit(n_splits = numSplits)
            for train_index, test_index in self.tss.split(self.df):
                self.dfTrainList.append(self.df.iloc[train_index, :])
                self.dfTestList.append(self.df.iloc[test_index, :])
                # change it!!!
                self.dataTrainList.append(self.dataFullNp[train_index])
                self.dataTestList.append(self.dataFullNp[test_index])
                self.dataTrainTestList.append(np.concatenate([self.dataFullNp[train_index], self.dataFullNp[test_index]]))
                
            self.setCurrentTrainTestData(0)
            self.setCurrentTrainTestDataNp(0)

    @classmethod
    def fromHowLong(cls, client, config, dataPair, howLong, interval, numSplits, onlyBinance = False):
        untilThisDate = datetime.datetime.now(datetime.timezone.utc)
        sinceThisDate = untilThisDate - datetime.timedelta(days = howLong)
        return cls(client, config, dataPair, sinceThisDate, untilThisDate, interval, numSplits, onlyBinance)
        
    def generateDbData(config, sinceThisDate, untilThisDate, interval):
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
                # If we want the data for the past hour with 1hr interval, we would need only 1 candle. Thus getting the time between open_time and close_time and not both open_time
                queryResult = pd.read_sql(f"select * from data_binance_btc_usdt where `interval` = '{interval}' and open_time >= '{sinceThisDate}' and close_time <= '{untilThisDate}'", mydb)
            finally:
                mydb.close()
            
        if len(queryResult) == 0:
            raise Exception(f"Cannot find appropriate data for start_time: {sinceThisDate} end_time: {untilThisDate} interval: {interval}")
    
        return queryResult
        
    def generateData(client, dataPair, sinceThisDate, untilThisDate, interval):        
        # Execute the query from binance - timestamps must be converted to strings
        candle = client.get_historical_klines(dataPair, interval, str(sinceThisDate), str(untilThisDate))

        # Create a dataframe to label all the columns returned by binance so we work with them later.
        df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])

        # as timestamp is returned in ms, let us convert this back to proper timestamps.
        dateTimeFormat = "%d-%m-%y %H:%M:%S"
        df.dateTime = pd.to_datetime(df.dateTime, unit='ms')#.dt.strftime(dateTimeFormat)
        df.closeTime = pd.to_datetime(df.closeTime, unit='ms')
        df.set_index('dateTime', inplace = True)
        df.sort_index(inplace = True)
        
        df.open = df.open.astype(float)
        df.close = df.close.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.volume = df.volume.astype(float)
        
        df[PRICE] = df["close"]

        # Get rid of columns we do not need
        df = df.drop(['quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
        return df
