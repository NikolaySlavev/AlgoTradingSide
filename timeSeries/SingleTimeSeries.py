from imports import *
from reporting.Reporting import Reporting


class SingleTimeSeries(ABC):
    def __init__(self, dataFullNp, dataTrainNp, dataTestNp, dataTrainTestNp, dataTrainWithWarmupNp, warmupTrainSize, warmupTestSize):
        self.dataFullNp = dataFullNp
        self.dataTrainNp = dataTrainNp
        self.dataTestNp = dataTestNp
        self.dataTrainTestNp = dataTrainTestNp
        self.dataTrainWithWarmupNp = dataTrainWithWarmupNp
        self.warmupTrainSize = warmupTrainSize
        self.warmupTestSize = warmupTestSize
        
        self.reports = Reporting()
        self.reportEnabled = False
    
    def getWarmupSize(self, useSet):
        if useSet == TRAIN:
            return self.warmupTrainSize
        elif useSet == TEST:
            return self.warmupTestSize
        elif useSet == TRAIN + WARMUP:
            return self.warmupTrainSize
        elif useSet == TEST + WARMUP:
            return self.warmupTestSize
        
        raise Exception(f"Invalid useSet {useSet}")
    
    def getPricesNp(self, useSet = TRAINTEST):
        dfNp = self.getSetNp(useSet)
        return dfNp[:, self.columnsNp[PRICE]]
        
    def getReturnsNp(self, useSet = TRAINTEST):
        dfNp = self.getSetNp(useSet)
        return dfNp[:, self.columnsNp[RETURN]]
        
    def getSetNp(self, useSet = ALL):
        if useSet == TRAIN:
            return self.dataTrainNp
        elif useSet == TEST:
            return self.dataTestNp
        elif useSet == TRAINTEST:
            return self.dataTrainTestNp
        elif useSet == ALL:
            return self.dataFullNp
        elif useSet == TRAIN + WARMUP:
            return self.dataTrainWithWarmupNp
        elif useSet == TEST + WARMUP:
            return self.dataTrainTestNp
        
        raise Exception("Invalid split of dataset")
