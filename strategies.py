import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg as AR
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

import arch
import warnings
import datetime
from abc import ABC, abstractmethod

from sklearn.model_selection import TimeSeriesSplit
from queue import PriorityQueue
from globals import *
import statistics_1

# filter some warnings
warnings.filterwarnings('ignore')


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


class TimeSeries(ABC):
    def __init__(self):
        self.df = pd.DataFrame()
        self.dfTrainTest = pd.DataFrame()
        self.dfTrain = pd.DataFrame()
        self.dfTest = pd.DataFrame()
        self.dfTrainList = []
        self.dfTestList = []
        self.dfTrainTestList = []
        self.reports = Reporting()
        self.reportEnabled = False
    
    @abstractmethod
    def generate_data():
        pass
    
    def get_df_col(self, col):
        return self.df[col]
    
    def get_prices(self, use_set = TRAINTEST):
        df = self.get_set(use_set)
        return df[PRICE]
    
    @staticmethod
    def get_static_prices(df):
        return df[PRICE]
    
    def pricesToReturns(prices):
        #return = prices.pct_change(1)
        if isinstance(prices, np.ndarray):
            return np.diff(prices) / prices[:-1]
            
        return (prices.shift(-1) / prices - 1).shift(1).fillna(0)
    
    def get_returns(self, use_set = TRAINTEST):
        if RETURN not in self.df.columns:
            return TimeSeries.pricesToReturns(self.get_prices(use_set))
        
        df = self.get_set(use_set)
        return df[RETURN]
    
    @staticmethod
    def get_static_returns(df):
        return df[RETURN]
    
    # Choose which data to use - train, test or both
    def get_set(self, use_set = ALL):
        if use_set == ALL:
            return self.df
        elif use_set == TRAINTEST:
            return self.dfTrainTest
        elif use_set == TRAIN:
            return self.dfTrain
        elif use_set == TEST:
            return self.dfTest
        
        raise Exception("Invalid split of dataaset")

    # Central plotting function to keep the plots consistent and save code repetition
    def plot(plotDataList, title, xlabel, ylabel, legendLoc = "upper left", axis = plt):
        for i in range(len(plotDataList)):
            axis.plot(plotDataList[i][1], label = plotDataList[i][0])
            
        axis.legend(loc = legendLoc, fontsize = 6)
        if type(axis) == type(plt):
            axis.title(title)
            axis.xlabel(xlabel)
            axis.ylabel(ylabel)
        else:
            axis.set_title(title)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            
    def set_current_train_test_data(self, index):
        self.dfTrain = self.dfTrainList[index]
        self.dfTest = self.dfTestList[index]
        self.dfTrainTest = pd.concat([self.dfTrain, self.dfTest])
    
    def set_current_train_data(self, index):
        self.dfTrain = self.dfTrainList[index]
        self.dfTrainTest = pd.concat([self.dfTrain, self.dfTest])    
    
    def set_current_test_data(self, index):
        self.dfTest = self.dfTestList[index]
        self.dfTrainTest = pd.concat([self.dfTrain, self.dfTest])
        
    def get_train_data(self, index):
        return self.dfTrainList[index]
        
    def get_test_data(self, index):
        return self.dfTestList[index]        
    
    
class BinanceTimeSeries(TimeSeries):
    def __init__(self, client, dataPair, howLong, interval, numSplits):
        super(BinanceTimeSeries, self).__init__()
        self.df = BinanceTimeSeries.generate_data(client, dataPair, howLong, interval)
        self.df[RETURN] = self.get_returns(use_set = ALL)
        self.tss = TimeSeriesSplit(n_splits = numSplits)
        for train_index, test_index in self.tss.split(self.df):
            self.dfTrainList.append(self.df.iloc[train_index, :])
            self.dfTestList.append(self.df.iloc[test_index, :])
        
        self.set_current_train_test_data(0)
        
    def generate_data(client, dataPair, howLong, interval):
        # Calculate the timestamps for the binance api function
        untilThisDate = datetime.datetime.now()
        sinceThisDate = untilThisDate - datetime.timedelta(days = howLong)
        
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


"""
    Class to create random Synthetic Time Series.
    The parameters are preset to meet the requirements, but can be adjusted if needed
    Seed can be specified to get the same Time Series
"""
class SyntheticTimeSeries(TimeSeries):
    # Create the Time Series
    def __init__(self, seed = 1, t = 3000, phi = 0.5, d = 0.02, theta = -0.3, mean = 0, variance = 1, p0 = 1000, p1 = 1000, train_test_split = 0.7):
        self.train_test_split = train_test_split
        self.df = SyntheticTimeSeries.generate_data(seed = seed, t = t, phi = phi, d = d, theta = theta, mean = mean, variance = variance, p0 = p0, p1 = p1)        
    
    def generate_data(seed, t, phi, d, theta, mean, variance, p0, p1):
        np.random.seed(seed)
        series = [p0, p1]
        change = p1 - p0
        eps =   np.random.normal(mean, variance, t)
        for i in range(1, t-1):
            change_prev = change
            change = phi * (change_prev - d) + eps[i] + theta * eps[i-1] + d                
            series.append(series[-1] + change)
        
        df = pd.DataFrame()
        df[PRICE] = pd.Series(data = series)
        return df
        

class Strategy():
    def __init__(self):
        self.useSet = TRAIN
        self.timeSeries = None
    
    # Computes SMA
    def get_simple_moving_average(prices, period):
        ma = prices.rolling(window = period).mean()
        
        # to avoid NA values in the first entries
        for i in range(period):
            ma[i] = prices[i]
        
        return ma

    # Computes EMA
    def get_exponential_moving_average(prices, alpha):
        return prices.ewm(alpha = alpha, adjust = False).mean()
    
    # Compute the Bands
    def get_bollinger_bands(prices, period = 20, num_std = 2):
        ma = Strategy.get_simple_moving_average(prices, period)
        std = prices.rolling(period).std() 
        upper = ma + num_std * std
        lower = ma - num_std * std
        for i in range(period):
            upper[i] = prices[i]
            lower[i] = prices[i]
           
        return upper, lower

    # Compute RSI
    def get_rsi(prices, period):
        delta = prices.diff()
        gain = delta.clip(lower = 0)
        loss = -1 * delta.clip(upper = 0)
        sma_gain = gain.rolling(period).mean()
        sma_loss = loss.rolling(period).mean()
        rs = sma_gain / sma_loss
        rsi = 100 - (100 / (rs + 1))
        rsi = rsi.fillna(50)
        return rsi
    
    def get_transaction_cost(signal, last_signal):
        if signal == BUY and last_signal != BUY or signal == SELL and last_signal != SELL:
            return TRANSACTION_COST
        
        return 0
    
    def get_simple_moving_average_ind(prices, last_price_index, period, weight):
        if last_price_index - period + 1 < 0:
            return prices[last_price_index]
        
        nominator = 0
        denominator = 0
        for i in range(period):
            nominator += weight * prices[last_price_index - i]
            denominator += weight
            
        result = nominator / denominator
        return result
    
    def position(price, signal, last_signal, prev_cash, prev_w):
        transaciton_cost = Strategy.get_transaction_cost(signal, last_signal)
        if signal == BUY:
            # BUY
            w = (prev_cash * (1 - transaciton_cost)) / price  + prev_w
            cash = 0
        elif signal == SELL:
            # SELL
            cash = prev_w * (1 - transaciton_cost) * price + prev_cash
            w = 0
        elif signal == HOLD:
            cash = prev_cash
            w = prev_w
        else:
            raise Exception("Wrong signal")
            
        return cash, w
    
    def getStrategyReturns(prices, signals, cashStart, useSet, startAfter):    
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        prev_cash = cashStart
        prev_w = 0
        last_signal = 0
        for i in range(len(prices)):
            cash[i], w[i] = Strategy.position(prices[i], signals[i], last_signal, prev_cash, prev_w)
            if signals[i] != HOLD:
                last_signal = signals[i]
                
            prev_cash = cash[i]
            prev_w = w[i]
        
        strategy_returns =  [a * b for a, b in zip(w, prices)] + cash
        strategy_returns = pd.Series(strategy_returns, index = prices.index)
        if useSet == TEST:
            strategy_returns = strategy_returns[startAfter:]
        
        return strategy_returns
    
    def setUseSet(self, useSet):
        self.useSet = useSet
        
    def getPrices(timeSeries, useSet):
        if useSet == TEST:
            return TimeSeries.get_static_prices(timeSeries.get_set(use_set = TRAINTEST))

        return TimeSeries.get_static_prices(timeSeries.get_set(use_set = useSet))
        
    def getStartAfter(timeSeries, useSet):
        return len(timeSeries.dfTrain) if useSet == TEST else 0
    
    def addReport(self, strategyName, strategyReturns, signals, params):
        if not self.timeSeries.reportEnabled:
            return
        
        df = self.timeSeries.get_set(use_set = self.useSet).copy()
        df["strategyReturns"] = strategyReturns
        df["signals"] = signals
        self.timeSeries.reports.addReport(strategyName = strategyName, params = params, finalCash = strategyReturns[-1], df = df)
    
 
"""
    Class that encapsulates the Trend Following Strategy
    Both with SMA and EMA was impelemnted for comparison purposes
    The df object needs to be of type SyntheticTimeSeries in order to reuse the code above
"""
class TrendFollowing(Strategy):
    def __init__(self, timeSeries, cashStart):
        if not isinstance(timeSeries, TimeSeries):
            return "Object needs to be an instance of TimeSeries"

        super(TrendFollowing, self).__init__()
        self.timeSeries = timeSeries
        self.cashStart = cashStart
    
    def signal(price, ma):
        if price > ma:
            return BUY
        elif price < ma:
            return SELL
        else:
            return HOLD
        
    def signal_crossover(ma_long, ma_short):
        if ma_short > ma_long:
            return BUY
        elif ma_short < ma_long:
            return SELL
        else:
            return HOLD
    
    def signal_bb_rsi(price, bb_upper, bb_lower, rsi):
        if rsi < 10 and price < bb_lower:
            return SELL
        elif rsi > 90 and price > bb_upper:
            return BUY
        else:
            return HOLD
    
    def TF_simple_bayes(self, period):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_data(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_simple(period)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
                
    def TF_simple(self, period):
        period = int(period)
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages = Strategy.get_simple_moving_average(prices, period)
            
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = TrendFollowing.signal(prices[i], moving_averages[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
            
        self.addReport("TFS", strategyReturns, signals, period)
        
        return strategyReturns
    
    def TF_simple_exec(prices, period):
        moving_averages = Strategy.get_simple_moving_average(prices, period)
        return TrendFollowing.signal(prices[-1], moving_averages[-1])
    
    def TF_exponential_bayes(self, alpha):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_data(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_exponential(alpha)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def TF_exponential(self, alpha):
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages = Strategy.get_exponential_moving_average(prices, alpha)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = TrendFollowing.signal(prices[i], moving_averages[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
            
        self.addReport("TFE", strategyReturns, signals, alpha)
        
        return strategyReturns
    
    def TF_bb_rsi_bayes(self, bb_period, bb_std, rsi_period):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_data(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_bb_rsi(bb_period, bb_std, rsi_period)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
        
    def TF_bb_rsi(self, bb_period, bb_std, rsi_period):
        bb_period = int(bb_period)
        rsi_period = int(rsi_period)
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        bb_upper, bb_lower = Strategy.get_bollinger_bands(prices, bb_period, bb_std)
        rsi = Strategy.get_rsi(prices, rsi_period)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = TrendFollowing.signal_bb_rsi(prices[i], bb_upper[i], bb_lower[i], rsi[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
            
        self.addReport("TFBBRSI", strategyReturns, signals, [bb_period, bb_std, rsi_period])
        
        return strategyReturns
    
    def TF_bb_rsi_exec(prices, bb_period, bb_std, rsi_period):
        bb_upper, bb_lower = Strategy.get_bollinger_bands(prices, bb_period, bb_std)
        rsi = Strategy.get_rsi(prices, rsi_period)
        return TrendFollowing.signal_bb_rsi(prices[-1], bb_upper[-1], bb_lower[-1], rsi[-1])
        
    def TF_crossover_simple_bayes(self, longPeriod, shortPeriod):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_data(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_crossover_simple(longPeriod, shortPeriod)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
        
    def TF_crossover_simple(self, longPeriod, shortPeriod):
        longPeriod = int(longPeriod)
        shortPeriod = int(shortPeriod)
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages_long = Strategy.get_simple_moving_average(prices, longPeriod)
        moving_averages_short = Strategy.get_simple_moving_average(prices, shortPeriod)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = TrendFollowing.signal_crossover(moving_averages_long[i], moving_averages_short[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
            
        self.addReport("TFCS", strategyReturns, signals, [longPeriod, shortPeriod])
        
        return strategyReturns
    
    def TF_crossover_exponential_bayes(self, longAlpha, shortAlpha):
        valuesOpt = []
        # if longAlpha >= shortAlpha:
        #     return -5
        
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_data(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_crossover_exponential(longAlpha, shortAlpha)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                                    
        return sum(valuesOpt) / len(valuesOpt)
    
    def TF_crossover_exponential(self, longAlpha, shortAlpha):
            prices = Strategy.getPrices(self.timeSeries, self.useSet)
            startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
            moving_averages_long = Strategy.get_exponential_moving_average(prices, longAlpha)
            moving_averages_short = Strategy.get_exponential_moving_average(prices, shortAlpha)
            
            signals = np.zeros(np.shape(prices))
            for i in range(0, len(prices)):
                if startAfter > i:
                    signals[i] = HOLD
                    continue
                
                signals[i] = TrendFollowing.signal_crossover(moving_averages_long[i], moving_averages_short[i])
                
            strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
            if self.useSet == TEST:
                signals = signals[startAfter:] 
            
            self.addReport("TFCE", strategyReturns, signals, [longAlpha, shortAlpha])
            
            return strategyReturns
        
    def TF_all(self, parameter):
        # ma_type -> simple, exponential
        # parameters -> for simple is period in range 0 to inf, for exponential is alpha in range 0 to 1
        # use_set -> trainTest, train, test, all
        
        prices = TimeSeries.get_static_prices(self.timeSeries.get_set(use_set = self.useSet)).reset_index(drop = True)
        startAfter = 0
        if self.useSet == TEST:
            startAfter = len(self.timeSeries.dfTrain)
            prices = TimeSeries.get_static_prices(self.timeSeries.get_set(use_set = TRAINTEST)).reset_index(drop = True)            
            
        if self.maType == "simple":
            parameter = int(parameter)
            moving_averages = Strategy.get_simple_moving_average(prices, parameter)
        elif self.maType == "exponential":
            moving_averages = Strategy.get_exponential_moving_average(prices, parameter)
        elif self.maType == "crossover_simple":
            moving_averages_long = Strategy.get_simple_moving_average(prices, parameter["long_period"])
            moving_averages_short = Strategy.get_simple_moving_average(prices, parameter["short_period"])
        elif self.maType == "crossover_exponential":
            moving_averages_long = Strategy.get_exponential_moving_average(prices, parameter["long_alpha"])
            moving_averages_short = Strategy.get_exponential_moving_average(prices, parameter["short_alpha"])
        elif self.maType == "bb_rsi":
            bb_upper, bb_lower = Strategy.get_bollinger_bands(prices, parameter["bb_period"], parameter["bb_std"])
            rsi = Strategy.get_rsi(prices, parameter["rsi_period"])
        else:
            raise Exception("Wrong maType")
        
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        signals = np.zeros(np.shape(prices))
        prev_cash = self.cash_start
        prev_w = 0
        last_signal = 0
        for i in range(0, len(prices) ):
            index = i + prices.first_valid_index()
            price = prices[index]
            #ma = Strategy.get_simple_moving_average(prices, index, sme_period, 1)
            
            if startAfter > i:
                signals[i] = HOLD
            elif self.maType == "simple" or self.maType == "exponential":
                signals[i] = TrendFollowing.signal(price, moving_averages[index])
            elif self.maType == "crossover_simple" or self.maType == "crossover_exponential":
                signals[i] = TrendFollowing.signal_crossover(moving_averages_long[index], moving_averages_short[index])
            elif self.maType == "bb_rsi":
                signals[i] = TrendFollowing.signal_bb_rsi(price, bb_upper[index], bb_lower[index], rsi[index])
            else:
                raise Exception("Wrong ma_type")
                
            cash[i], w[i] = Strategy.position(price, signals[i], last_signal, prev_cash, prev_w)
            if signals[i] != HOLD:
                last_signal = signals[i]
                
            prev_cash = cash[i]
            prev_w = w[i]
        
        strategy_returns =  [a * b for a, b in zip(w, prices)] + cash
        if self.useSet == TEST:
            strategy_returns = strategy_returns[startAfter:]
            signals = signals[startAfter:]
        
        df = self.timeSeries.get_set(use_set = self.useSet)
        df["strategyReturns"] = strategy_returns
        df["signals"] = signals
        self.timeSeries.reports.addReport(strategyName = "TF" + self.maType, params = parameter, finalCash = strategy_returns[-1], df = df)
        
        return strategy_returns[-1]
        
    # Trend Following with simple moving average
    def TF_all_old(self, ma_type, parameter, use_set = TRAINTEST):
        # ma_type -> simple, exponential
        # parameters -> for simple is period in range 0 to inf, for exponential is alpha in range 0 to 1
        # use_set -> trainTest, train, test, all
        
        prices = TimeSeries.get_static_prices(self.timeSeries.get_set(use_set = use_set)).reset_index(drop = True)
        startAfter = 0
        if use_set == TEST:
            startAfter = len(self.timeSeries.dfTrain)
            prices = TimeSeries.get_static_prices(self.timeSeries.get_set(use_set = TRAINTEST)).reset_index(drop = True)            
            
        if ma_type == "simple":
            moving_averages = Strategy.get_simple_moving_average(prices, parameter)
        elif ma_type == "exponential":
            moving_averages = Strategy.get_exponential_moving_average(prices, parameter)
        elif ma_type == "crossover_simple":
            moving_averages_long = Strategy.get_simple_moving_average(prices, parameter["long_period"])
            moving_averages_short = Strategy.get_simple_moving_average(prices, parameter["short_period"])
        elif ma_type == "crossover_exponential":
            moving_averages_long = Strategy.get_exponential_moving_average(prices, parameter["long_alpha"])
            moving_averages_short = Strategy.get_exponential_moving_average(prices, parameter["short_alpha"])
        elif ma_type == "bb_rsi":
            bb_upper, bb_lower = Strategy.get_bollinger_bands(prices, parameter["bb_period"], parameter["bb_std"])
            rsi = Strategy.get_rsi(prices, parameter["rsi_period"])
        else:
            raise Exception("Wrong ma_type")
        
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        signals = np.zeros(np.shape(prices))
        prev_cash = self.cash_start
        prev_w = 0
        last_signal = 0
        for i in range(0, len(prices) ):
            index = i + prices.first_valid_index()
            price = prices[index]
            #ma = Strategy.get_simple_moving_average(prices, index, sme_period, 1)
            
            if startAfter > i:
                signals[i] = HOLD
            elif ma_type == "simple" or ma_type == "exponential":
                signals[i] = TrendFollowing.signal(price, moving_averages[index])
            elif ma_type == "crossover_simple" or ma_type == "crossover_exponential":
                signals[i] = TrendFollowing.signal_crossover(moving_averages_long[index], moving_averages_short[index])
            elif ma_type == "bb_rsi":
                signals[i] = TrendFollowing.signal_bb_rsi(price, bb_upper[index], bb_lower[index], rsi[index])
            else:
                raise Exception("Wrong ma_type")
                
            cash[i], w[i] = Strategy.position(price, signals[i], last_signal, prev_cash, prev_w)
            if signals[i] != HOLD:
                last_signal = signals[i]
                
            prev_cash = cash[i]
            prev_w = w[i]
        
        strategy_returns =  [a * b for a, b in zip(w, prices)] + cash
        if use_set == TEST:
            strategy_returns = strategy_returns[startAfter:]
            signals = signals[startAfter:]
        
        df = self.timeSeries.get_set(use_set = use_set)
        df["strategyReturns"] = strategy_returns
        df["signals"] = signals
        self.timeSeries.reports.addReport(strategyName = "TF" + ma_type, params = parameter, finalCash = strategy_returns[-1], df = df)
        
        return strategy_returns
        

"""
    Class that encapsulates the Mean Reversion Strategy
    Implemented both SMA and Bands+RSI
"""
class MeanReversion(Strategy):
    def __init__(self, timeSeries, cashStart):
        if not isinstance(timeSeries, TimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        super(MeanReversion, self).__init__()
        self.timeSeries = timeSeries
        self.cashStart = cashStart
    
    def signal(price, ma):
        if ma > price: 
            return BUY
        elif ma < price: 
            return SELL
        else:
            return HOLD
    
    def signal_crossover(ma_long, ma_short):
        if ma_short < ma_long:
            return BUY
        elif ma_short > ma_long:
            return SELL
        else:
            return HOLD
        
    def signal_bb_rsi(price, bb_upper, bb_lower, rsi):
        if rsi < 10 and price < bb_lower:
            return BUY
        elif rsi > 90 and price > bb_upper:
            return SELL
        else:
            return HOLD
    
    def MR_simple_bayes(self, period):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_test_data(i)
            self.setUseSet(TRAIN)
            valuesOpt.append(self.MR_simple(period)[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def MR_exponential_bayes(self, alpha):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_test_data(i)
            self.setUseSet(TRAIN)
            valuesOpt.append(self.MR_exponential(alpha)[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def MR_crossover_simple_bayes(self, longPeriod, shortPeriod):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_test_data(i)
            self.setUseSet(TRAIN)
            valuesOpt.append(self.MR_crossover_simple(longPeriod, shortPeriod)[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def MR_crossover_exponential_bayes(self, longAlpha, shortAlpha):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_test_data(i)
            self.setUseSet(TRAIN)
            valuesOpt.append(self.MR_crossover_exponential(longAlpha, shortAlpha)[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def MR_bb_rsi_bayes(self, bb_period, bb_std, rsi_period):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.set_current_train_test_data(i)
            self.setUseSet(TRAIN)
            valuesOpt.append(self.MR_bb_rsi(bb_period, bb_std, rsi_period)[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def MR_simple(self, period):
        period = int(period)
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages = Strategy.get_simple_moving_average(prices, period)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = MeanReversion.signal(prices[i], moving_averages[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
                
        self.addReport("MRS", strategyReturns, signals, period)
        
        return strategyReturns
    
    def MR_exponential_exec(prices, alpha):
        moving_averages = Strategy.get_exponential_moving_average(prices, alpha)
        return MeanReversion.signal(prices[-1], moving_averages[-1])
    
    def MR_exponential(self, alpha):
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages = Strategy.get_exponential_moving_average(prices, alpha)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = MeanReversion.signal(prices[i], moving_averages[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
        
        self.addReport("MRE", strategyReturns, signals, alpha)
        
        return strategyReturns
        
    def MR_bb_rsi(self, bb_period, bb_std, rsi_period):
        bb_period = int(bb_period)
        rsi_period = int(rsi_period)
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        bb_upper, bb_lower = Strategy.get_bollinger_bands(prices, bb_period, bb_std)
        rsi = Strategy.get_rsi(prices, rsi_period)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = MeanReversion.signal_bb_rsi(prices[i], bb_upper[i], bb_lower[i], rsi[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
        
        self.addReport("MRBR", strategyReturns, signals, [bb_period, bb_std, rsi_period])
        
        return strategyReturns
        
    def MR_crossover_simple(self, longPeriod, shortPeriod):
        longPeriod = int(longPeriod)
        shortPeriod = int(shortPeriod)
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages_long = Strategy.get_simple_moving_average(prices, longPeriod)
        moving_averages_short = Strategy.get_simple_moving_average(prices, shortPeriod)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = MeanReversion.signal_crossover(moving_averages_long[i], moving_averages_short[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        self.addReport("MRCS", strategyReturns, signals, [longPeriod, shortPeriod])
        
        return strategyReturns
    
    def MR_crossover_exponential(self, longAlpha, shortAlpha):
        prices = Strategy.getPrices(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages_long = Strategy.get_exponential_moving_average(prices, longAlpha)
        moving_averages_short = Strategy.get_exponential_moving_average(prices, shortAlpha)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = MeanReversion.signal_crossover(moving_averages_long[i], moving_averages_short[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        self.addReport("MRCE", strategyReturns, signals, [longAlpha, shortAlpha])
        
        return strategyReturns
    
    # Mean Reversion with SMA
    def MR_all(self, ma_type, parameter, use_set = TRAINTEST):
        prices = TimeSeries.get_static_prices(self.timeSeries.get_set(use_set = use_set)).reset_index(drop = True)
        startAfter = 0
        if use_set == TEST:
            startAfter = len(self.timeSeries.dfTrain)
            prices = TimeSeries.get_static_prices(self.timeSeries.get_set(use_set = TRAINTEST)).reset_index(drop = True)            
            
        if ma_type == "simple":
            moving_averages = Strategy.get_simple_moving_average(prices, parameter)
        elif ma_type == "crossover_simple":
            moving_averages_long = Strategy.get_simple_moving_average(prices, parameter["long_period"])
            moving_averages_short = Strategy.get_simple_moving_average(prices, parameter["short_period"])
        elif ma_type == "crossover_exponential":
            moving_averages_long = Strategy.get_exponential_moving_average(prices, parameter["long_alpha"])
            moving_averages_short = Strategy.get_exponential_moving_average(prices, parameter["short_alpha"])
        elif ma_type == "bb_rsi":
            bb_upper, bb_lower = Strategy.get_bollinger_bands(prices, parameter["bb_period"], parameter["bb_std"])
            rsi = Strategy.get_rsi(prices, parameter["rsi_period"])
        else:
            raise Exception("Wrong ma_type")
        
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        signals = np.zeros(np.shape(prices))
        
        last_signal = 0
        prev_cash = self.cash_start
        prev_w = 0
        for i in range(len(prices)):
            index = i + prices.first_valid_index()
            price = prices[index]
            
            if startAfter > i:
                signals[i] = HOLD
            elif ma_type == "simple":
                signals[i] = MeanReversion.signal(price, moving_averages[index])
            elif ma_type == "crossover_simple" or ma_type == "crossover_exponential":
                signals[i] = TrendFollowing.signal_crossover(moving_averages_long[index], moving_averages_short[index])
            elif ma_type == "bb_rsi":
                signals[i] = MeanReversion.signal_bb_rsi(price, bb_upper[index], bb_lower[index], rsi[index])
                            
            cash[i], w[i] = Strategy.position(price, signals[i], last_signal, prev_cash, prev_w)
            if signals[i] != HOLD:
                last_signal = signals[i]
            
            prev_cash = cash[i]
            prev_w = w[i]
        
        strategy_returns =  [a * b for a, b in zip(w, prices)] + cash                
        if use_set == TEST:
            strategy_returns = strategy_returns[startAfter:]
            signals = signals[startAfter:]
            
        df = self.timeSeries.get_set(use_set = use_set)
        df["strategyReturns"] = strategy_returns
        df["signals"] = signals
        self.timeSeries.reports.addReport(strategyName = "TF" + ma_type, params = parameter, finalCash = strategy_returns[-1], df = df)
            
        return strategy_returns


"""
    Class that encapsulates ARIMA+GARCH trading strategy
"""
class ForecastArimaGarch(Strategy):    
    def __init__(self, timeSeries, cash_start):
        if not isinstance(timeSeries, TimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        self.timeSeries = timeSeries
        self.cash_start = cash_start
        self.prices = self.timeSeries.get_prices()
        self.returnsTrain = TimeSeries.get_static_returns(self.timeSeries.dfTrain)
        self.returnsTest = TimeSeries.get_static_returns(self.timeSeries.dfTest)
        self.returns = self.returnsTrain.append(self.returnsTest)
        
    # Make one ARIMA+GARCH prediction
    def ARIMA_predict(data, p, d, q):
        # fit ARIMA on returns
        arima_model = ARIMA(data, order=(p,d,q))
        model_fit = arima_model.fit()
        arima_residuals = model_fit.resid

        # fit a GARCH(1,1) model on the residuals of the ARIMA model
        garch = arch.arch_model(arima_residuals, p=1, q=1)
        garch_fitted = garch.fit(disp=False)

        # Use ARIMA to predict mu
        predicted_mu = model_fit.forecast()
        
        # Use GARCH to predict the residual
        garch_forecast = garch_fitted.forecast(horizon=1)
        predicted_et = garch_forecast.mean['h.1'].iloc[-1]

        # Combine both models' output: yt = mu + et
        return predicted_mu + predicted_et

    # Buy, hold or sell given the arima prediciton
    def ARIMA_position(arima_prediction, cash, w, price):
        if arima_prediction == 0:
            next_w = w
            next_cash = cash
            buySell = 0
        elif arima_prediction > 0:
            # BUY
            next_w = cash / price  + w
            next_cash = 0
            buySell = 1
        else:
            # SELL
            next_cash = w * price + cash
            next_w = 0
            buySell = -1
            
        return next_cash, next_w, buySell
    
    # ARIMA+GARCH only on the test set
    def ARIMA_GARCH_test(self):
        arima_prediction = np.zeros(np.shape(self.returns))
        w = np.zeros(np.shape(self.returns))
        cash = np.zeros(np.shape(self.returns))
        cash[0] = self.cash_start
            
        auto_arima_model = pm.auto_arima(self.returnsTrain, start_p = 1, start_q = 1, max_p = 20, max_q = 20, trace = False)        
        p, d, q = auto_arima_model.order
        print(auto_arima_model.summary())
        
        buySellPos = np.zeros(np.shape(self.returns))
        for i in range(len(self.returns) - 1):
            arima_prediction[i] = 0            
            if i >= len(self.returnsTrain):
                X = self.returns[0:i]
                train = X
                arima_prediction[i] = ForecastArimaGarch.ARIMA_predict(train, p, d, q)       
            
            cash[i+1], w[i+1], buySellPos[i+1] = ForecastArimaGarch.ARIMA_position(arima_prediction[i], cash[i], w[i], self.prices[i])
                
        arima_strategy = [a*b for a,b in zip(w, self.prices)] + cash
        arima_strategy = arima_strategy[len(self.returnsTrain):]
        arima_prediction = arima_prediction[len(self.returnsTrain):]
        buySellPos = buySellPos[len(self.returnsTrain):]
        return arima_strategy, arima_prediction, buySellPos, str(auto_arima_model.order)
    
    # ARIMA+GARCH on the whole dataset
    def ARIMA_GARCH_all(self):
        arima_prediction = np.zeros(np.shape(self.returnsTrain))
        w = np.zeros(np.shape(self.returnsTrain))
        cash = np.zeros(np.shape(self.returnsTrain))
        cash[0] = self.cash_start
        
        auto_arima_model = pm.auto_arima(self.returnsTrain, start_p=1, start_q=1, max_p=20, max_q=20, trace = False)
        predictions_in_sample = auto_arima_model.predict_in_sample()
        p, d, q = auto_arima_model.order
        
        for i in range(len(self.returns) - 1):
            if i >= len(self.returnsTrain):
                X = self.returns[0:i]
                train = X
                arima_prediction[i] = ForecastArimaGarch.ARIMA_predict(train, p, d, q)       
            else:
                arima_prediction[i] = predictions_in_sample[i]
            
            cash[i+1], w[i+1], _ = ForecastArimaGarch.ARIMA_position(arima_prediction[i], cash[i], w[i], self.prices[i])
                
        arima_strategy = [a*b for a,b in zip(self.w, self.prices)] + self.cash
        return arima_strategy, arima_prediction, str(auto_arima_model.order)
    
    # ARIMA+GARCH only on the train set
    def ARIMA_GARCH_train(self):
        arima_prediction = np.zeros(np.shape(self.returnsTrain))
        w = np.zeros(np.shape(self.returnsTrain))
        cash = np.zeros(np.shape(self.returnsTrain))
        cash[0] = self.cash_start
        
        auto_arima_model = pm.auto_arima(self.returnsTrain, start_p = 1, start_q = 1, max_p = 20, max_q = 20, trace = False)
        arima_prediction = auto_arima_model.predict_in_sample()
        for i in range(len(arima_prediction) - 1):
            cash[i+1], w[i+1], _ = ForecastArimaGarch.ARIMA_position(arima_prediction[i], cash[i], w[i], self.prices[i])
                
        arima_strategy = [a*b for a,b in zip(w, self.prices)] + cash
        return arima_strategy, arima_prediction, str(auto_arima_model.order)
