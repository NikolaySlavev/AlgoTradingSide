from imports import *
from timeSeries.TimeSeries import TimeSeries


class BinanceTimeSeries(TimeSeries):
    def __init__(self, client, dataPair, sinceThisDate, untilThisDate, interval, numSplits):
        super(BinanceTimeSeries, self).__init__()
        self.df = BinanceTimeSeries.generateData(client, dataPair, sinceThisDate, untilThisDate, interval)
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
    def fromHowLong(cls, client, dataPair, howLong, interval, numSplits):
        untilThisDate = datetime.datetime.now()
        sinceThisDate = untilThisDate - datetime.timedelta(days = howLong)
        return cls(client, dataPair, sinceThisDate, untilThisDate, interval, numSplits)
        
    def generateData(client, dataPair, sinceThisDate, untilThisDate, interval):        
        # Execute the query from binance - timestamps must be converted to strings
        candle = client.get_historical_klines(dataPair, interval, str(sinceThisDate), str(untilThisDate))

        # Create a dataframe to label all the columns returned by binance so we work with them later.
        df = pd.DataFrame(candle, columns=['dateTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol', 'takerBuyQuoteVol', 'ignore'])

        # as timestamp is returned in ms, let us convert this back to proper timestamps.
        dateTimeFormat = "%d-%m-%y %H:%M:%S"
        df.dateTime = pd.to_datetime(df.dateTime, unit='ms')#.dt.strftime(dateTimeFormat)
        df.set_index('dateTime', inplace = True)
        df.sort_index(inplace = True)
        
        df.open = df.open.astype(float)
        df.close = df.close.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        
        df[PRICE] = df["close"]

        # Get rid of columns we do not need
        df = df.drop(['closeTime', 'quoteAssetVolume', 'numberOfTrades', 'takerBuyBaseVol','takerBuyQuoteVol', 'ignore'], axis=1)
        return df
