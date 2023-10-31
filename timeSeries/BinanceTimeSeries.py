from imports import *
from timeSeries.TimeSeries import TimeSeries
from timeSeries.SingleTimeSeries import SingleTimeSeries

import sshtunnel
import MySQLdb


renameDbData = {"open_time": "dateTime", "close_time": "closeTime" , "open_price": "open", "close_price": "close", "high_price": "high", "low_price": "low"}

class BinanceTimeSeries(TimeSeries):
    def __init__(self, client, config, dataPair, sinceThisDate, untilThisDate, interval, numSplits, onlyBinance = False, splitType = TimeSeriesSplitTypes.NORMAL, initialWarmupData = []):
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
            self.df = pd.concat([self.dfDb, self.dfBinance])
        
        self.df[RETURN] = self.getReturns(useSet = ALL)
        self.dataFullNp = TimeSeries.dfToNp(self.df)
        self.columnsNp = TimeSeries.dfColumnsToNp(self.df)
        
        if splitType == TimeSeriesSplitTypes.SKLEARN:
            self.singleTimeSeriesList = BinanceTimeSeries.getSklearnSplit(self.df, self.dataFullNp, initialWarmupData, numSplits)
        elif splitType == TimeSeriesSplitTypes.NORMAL:
             self.singleTimeSeriesList= BinanceTimeSeries.getNormalSplit(self.dataFullNp, initialWarmupData, numSplits)
        elif splitType == TimeSeriesSplitTypes.NONE:
            self.singleTimeSeriesList = BinanceTimeSeries.getNoneSplit(self.dataFullNp, initialWarmupData)
        else:
            raise Exception(f"Invalid splitType {splitType}")

        self.setCurrentSingleTimeSeries(0)
            
    @classmethod
    def fromHowLong(cls, client, config, dataPair, howLong, interval, numSplits, onlyBinance = False, splitType = TimeSeriesSplitTypes.NORMAL, initialWarmupData = None):
        untilThisDate = datetime.datetime.now(datetime.timezone.utc)
        sinceThisDate = untilThisDate - datetime.timedelta(days = howLong)
        return cls(client, config, dataPair, sinceThisDate, untilThisDate, interval, numSplits, onlyBinance, splitType, initialWarmupData)
        
    def getCustomSplit(df, dataFullNp, customSplits: dict):
        # FINISH IT
        dataTrainList = []
        dataTestList = []
        dataTrainTestList = []
        
        cut = int(len(df) * 0.7)
        dataTrainList.append(dataFullNp[:cut])
        dataTestList.append(dataFullNp[cut:])
        dataTrainTestList.append(np.concatenate([dataFullNp[:cut], dataFullNp[cut:]]))
        
    def getNoneSplit(dataFullNp, initialWarmupData):
        return [SingleTimeSeries(dataFullNp = dataFullNp, 
                                 dataTrainNp = dataFullNp, 
                                 dataTestNp = dataFullNp, 
                                 dataTrainTestNp = np.concatenate([initialWarmupData, dataFullNp]) if len(initialWarmupData) != 0 else dataFullNp,
                                 dataTrainWithWarmupNp = np.concatenate([initialWarmupData, dataFullNp]) if len(initialWarmupData) != 0 else dataFullNp,
                                 warmupTrainSize = len(initialWarmupData),
                                 warmupTestSize = len(initialWarmupData))]
    
    def getNormalSplit(dataFullNp, initialWarmupData, numSplits):
        singleTimeSeriesList = []
        splitSize = len(dataFullNp) // numSplits
        for i in range(0, numSplits - 1):
            dataTrainNp = dataFullNp[i * splitSize : (i + 1) * splitSize]
            dataTestNp = dataFullNp[(i + 1) * splitSize : (i + 2) * splitSize]
            warmupDf = initialWarmupData if i == 0 else singleTimeSeriesList[-1].dataTrainNp
            singleTimeSeries = SingleTimeSeries(dataFullNp = dataFullNp,
                                                dataTrainNp = dataTrainNp,
                                                dataTestNp = dataTestNp,
                                                dataTrainTestNp = np.concatenate([dataTrainNp, dataTestNp]),
                                                dataTrainWithWarmupNp = np.concatenate([warmupDf, dataTrainNp]),
                                                warmupTrainSize = len(warmupDf),
                                                warmupTestSize = len(dataTrainNp))
            singleTimeSeriesList.append(singleTimeSeries)
        
        # len(dataFullNp) / numSplits might not be exact so we need to take the matrix until the end and the size of the test set is >= of each size of the train set
        dataLastTestNp = dataFullNp[(numSplits - 1) * splitSize :]
        singleTimeSeriesList[-1].dataTestNp = dataLastTestNp
        singleTimeSeriesList[-1].dataTrainTestNp = np.concatenate([singleTimeSeriesList[-1].dataTrainNp, dataLastTestNp])
        
        return singleTimeSeriesList
        
    def getSklearnSplit(df, dataFullNp, initialWarmupData, numSplits):
        singleTimeSeriesList = []        
        # if numSplits = 10, sklearn splits the data into 10 training sets (i.e. divides len(df) by 10 + 1)
        tss = TimeSeriesSplit(n_splits = numSplits - 1)
        # try to pass the np matrix directly (dataFullNp) instead of the df
        for train_index, test_index in tss.split(df):
            singleTimeSeries = SingleTimeSeries(dataFullNp = dataFullNp, 
                                                dataTrainNp = dataFullNp[train_index],
                                                dataTestNp = dataFullNp[test_index], 
                                                dataTrainTestNp = np.concatenate([dataFullNp[train_index], dataFullNp[test_index]]),
                                                dataTrainWithWarmupNp = np.concatenate([initialWarmupData, dataFullNp[train_index]]),
                                                warmupTrainSize = len(initialWarmupData),
                                                warmupTestSize = len(dataFullNp[train_index]))
            singleTimeSeriesList.append(singleTimeSeries)
            
        return singleTimeSeriesList

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
        df.dateTime = pd.to_datetime(df.dateTime, unit = "ms")
        df.closeTime = pd.to_datetime(df.closeTime, unit = "ms")
        df.set_index("dateTime", inplace = True)
        df.sort_index(inplace = True)
        
        df.open = df.open.astype(float)
        df.close = df.close.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.volume = df.volume.astype(float)
        
        df[PRICE] = df["close"]

        # Get rid of columns we do not need
        df = df.drop(["quoteAssetVolume", "numberOfTrades", "takerBuyBaseVol", "takerBuyQuoteVol", "ignore"], axis = 1)
        return df
