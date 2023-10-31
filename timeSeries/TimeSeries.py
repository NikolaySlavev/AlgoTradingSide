from imports import *
from reporting.Reporting import Reporting
from timeSeries.SingleTimeSeries import SingleTimeSeries

class TimeSeries(ABC):
    def __init__(self):
        self.df = pd.DataFrame()
        
        # NP
        self.columnsNp = np.array([])
        
        self.singleTimeSeriesList = []
        self.singleTimeSeries = None
        
        self.reports = Reporting()
        self.reportEnabled = False
    
    def dfToNp(df):
        dfR = df.reset_index()
        return dfR.to_numpy(dtype = np.float64)
    
    def dfColumnsToNp(df):
        dfR = df.reset_index()
        columns = dfR.columns.to_numpy()
        columnsNp = {}
        for i in range(len(columns)):
            columnsNp[columns[i]] = i
            
        return columnsNp

    def pricesToReturns(prices):
        #return = prices.pct_change(1)
        if isinstance(prices, np.ndarray):
            return np.diff(prices) / prices[:-1]
            
        return (prices.shift(-1) / prices - 1).shift(1).fillna(0)

    def getWarmupSize(self, useSet):
        return self.singleTimeSeries.getWarmupSize(useSet)

    def getReturns(self, useSet = TRAINTEST):
        if RETURN not in self.df.columns:
            return TimeSeries.pricesToReturns(self.getPrices(useSet))
        
        df = self.get_set(useSet)
        return df[RETURN]

    def getPrices(self, useSet = TRAINTEST):
        df = self.getSet(useSet)
        return df[PRICE]

    def getDates(self, useSet):
        dfNp = self.getSetNp(useSet)
        return dfNp[:, self.columnsNp[DATETIME]]

    def getPricesNp(self, useSet = TRAINTEST):
        dfNp = self.getSetNp(useSet)
        return dfNp[:, self.columnsNp[PRICE]]
        
    def getReturnsNp(self, useSet = TRAINTEST):
        dfNp = self.getSetNp(useSet)
        return dfNp[:, self.columnsNp[RETURN]]
    
    # Choose which data to use - train, test or both
    def getSet(self, useSet = ALL):
        if useSet == ALL:
            return self.df
        elif useSet == TRAINTEST:
            return self.dfTrainTest
        elif useSet == TRAIN:
            return self.dfTrain
        elif useSet == TEST:
            return self.dfTest
        
        raise Exception("Invalid split of dataset")
    
    def getSetNp(self, useSet = ALL):
        return self.singleTimeSeries.getSetNp(useSet)

    def setCurrentSingleTimeSeries(self, index):
        self.singleTimeSeries = self.singleTimeSeriesList[index]

    # Central plotting function to keep the plots consistent and save code repetition
    def plot(plotDataList, title = "Title", xlabel = "X", ylabel = "Y", legendLoc = "upper left", xValues = None, axis = plt):
        for i in range(len(plotDataList)):
            if xValues is None:
                axis.plot(plotDataList[i][1], label = plotDataList[i][0])
            else:
                axis.plot(xValues, plotDataList[i][1], label = plotDataList[i][0])
            
        axis.legend(loc = legendLoc, fontsize = 6)
        if type(axis) == type(plt):
            axis.title(title)
            axis.xlabel(xlabel)
            axis.ylabel(ylabel)
        else:
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
                
    def plotPrices(self, useSet):
        pricesNp = self.getPricesNp(useSet)
        plt.plot(pricesNp, label = useSet)
        plt.title(f"Prices for {useSet} data")
        plt.show()
        
    def plotIndicators(self, useSet, indicatorDf):
        pricesNp = self.getPricesNp(useSet)
        dates = pd.to_datetime(self.getDates(useSet))
        if not indicatorDf['dates'].equals(pd.Series(dates)):
            raise Exception("Dates not equal")
        
        # plotting prices
        plt.plot(pricesNp, label = useSet)
        # plotting indicators
        plt.plot(indicatorDf, label = "indicator")
        plt.show()
        
    def plotIndicatorsAndSignals():
        pass
        
        
        
