from imports import *
from reporting.Reporting import Reporting


class TimeSeries(ABC):
    def __init__(self):
        # NP
        self.dataFullNp = np.array([])
        self.columnsNp = np.array([])
        
        # need to add that function df to np and call it after generateData
        self.dataTrainList = []
        self.dataTestList = []
        self.dataTrainTestList = []
        
        self.trainIndex = 0
        self.testIndex = 0
        self.dataTrainNp = np.array([]) 
        self.dataTestNp = np.array([])
        self.dataTrainTestNp = np.array([])
        
        # PD
        self.df = pd.DataFrame()
        self.dfTrainTest = pd.DataFrame()
        self.dfTrain = pd.DataFrame()
        self.dfTest = pd.DataFrame()
        self.dfTrainList = []
        self.dfTestList = []
        self.dfTrainTestList = []
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
    
    @abstractmethod
    def generateData():
        pass
    
    def getDfCol(self, col):
        return self.df[col]
    
    def getPrices(self, useSet = TRAINTEST):
        df = self.getSet(useSet)
        return df[PRICE]
    
    def getPricesNp(self, useSet = TRAINTEST):
        dfNp = self.getSetNp(useSet)
        return dfNp[:, self.columnsNp[PRICE]]#.astype(np.float64)
    
    @staticmethod
    def getStaticPrices(df):
        return df[PRICE]
    
    def pricesToReturns(prices):
        #return = prices.pct_change(1)
        if isinstance(prices, np.ndarray):
            return np.diff(prices) / prices[:-1]
            
        return (prices.shift(-1) / prices - 1).shift(1).fillna(0)

    def getReturnsNp(self, useSet = TRAINTEST):
        dfNp = self.getSetNp(useSet)
        return dfNp[:, self.columnsNp[RETURN]]
    
    def getReturns(self, useSet = TRAINTEST):
        if RETURN not in self.df.columns:
            return TimeSeries.pricesToReturns(self.getPrices(useSet))
        
        df = self.get_set(useSet)
        return df[RETURN]
    
    @staticmethod
    def getStaticReturns(df):
        return df[RETURN]
    
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
        if useSet == TRAIN:
            return self.dataTrainNp
        elif useSet == TEST:
            return self.dataTestNp
        elif useSet == TRAINTEST:
            return self.dataTrainTestNp
        elif useSet == ALL:
            return self.dataFullNp
        
        raise Exception("Invalid split of dataset")

    # Central plotting function to keep the plots consistent and save code repetition
    def plot(plotDataList, title, xlabel, ylabel, legendLoc = "upper left", xValues = None, axis = plt):
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
            
    def setCurrentTrainTestData(self, index):
        self.dfTrain = self.dfTrainList[index]
        self.dfTest = self.dfTestList[index]
        self.dfTrainTest = pd.concat([self.dfTrain, self.dfTest])
    
    def setCurrentTrainData(self, index):
        self.dfTrain = self.dfTrainList[index]
        self.dfTrainTest = pd.concat([self.dfTrain, self.dfTest])    
    
    def setCurrentTestData(self, index):
        self.dfTest = self.dfTestList[index]
        self.dfTrainTest = pd.concat([self.dfTrain, self.dfTest])
        
    def getTrainData(self, index):
        return self.dfTrainList[index]
        
    def getTestData(self, index):
        return self.dfTestList[index]
    
    def setCurrentTrainTestDataNp(self, index):
        self.dataTrainNp = self.dataTrainList[index]
        self.dataTestNp = self.dataTestList[index]
        self.dataTrainTestNp = self.dataTrainTestList[index]
    
    def setCurrentTrainDataNp(self, index):
        self.dataTrainNp = self.dataTrainList[index]
        self.dataTrainTestNp = self.dataTrainTestList[index]
    
    def setCurrentTestDataNp(self, index):
        self.dataTestNp = self.dataTestList[index]
        self.dataTrainTestNp = self.dataTrainTestList[index]
