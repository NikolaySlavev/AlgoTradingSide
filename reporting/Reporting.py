from imports import *

class Reporting():
    # the series separate or straight pass the full df?
    def __init__(self):
        self.content = {}
        self.q = {}
        self.maxLength = 3
    
    def addReport(self, strategyName, params, finalCash, df = pd.DataFrame(columns = ["price", "return", "cash", "signal"])):
        key = str(strategyName) + "_" + str(round(finalCash, 2)) + "_" + str(params)
        self.content[key] = df
        
        if strategyName not in self.q.keys():
            self.q[strategyName] = PriorityQueue()

        self.q[strategyName].put((finalCash, key))
        if self.q[strategyName].qsize() > self.maxLength:
            _, keyPop = self.q[strategyName].get()
            del self.content[keyPop]
        
    def writeReport(self, path):                
        for key in self.content:
            keyPath = key.replace(":", "")
            self.content[key].to_csv(path + keyPath + ".csv")
    
    def writeReportTest(self, path):
        if len(self.content) == 0:
            return 
        
        df = self.content[next(iter(self.content))]
        for key in self.content:
            df["strategyReturn " + str(key)] = self.content[key]["strategyReturns"]
            df["signals " + str(key)] = self.content[key]["signals"]
            
        df.to_csv(path + "reportTest.csv")
    
    def cleanVars(self):
        self.content = {}
        self.q = {}
