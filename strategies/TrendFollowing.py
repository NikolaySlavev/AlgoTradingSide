from imports import *
from strategies.Strategy import Strategy
from timeSeries.TimeSeries import TimeSeries


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
        else:
            return SELL
    
    @numba.jit((numba.float64[:], numba.float64[:], numba.int64), nopython = True, nogil = True)
    def getSignals(prices, mas, startAfter):
        signals = np.zeros(np.shape(prices))
        for i in range(startAfter, len(prices)):
            if prices[i] > mas[i]:
                signals[i] = BUY
            else:
                signals[i] =  SELL
                
        return signals
        
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
        for i in range(len(self.timeSeries.dataTrainList)):
            #self.timeSeries.setCurrentTrainData(i)
            self.timeSeries.setCurrentTrainDataNp(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_simple(period)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
                
    def TF_simple(self, period):
        period = int(period)
        prices = Strategy.getPricesNp(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        movingAverages = Strategy.getSimpleMovingAverageNp(prices, period)
        signals = TrendFollowing.getSignals(prices, movingAverages, startAfter)
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:]
            
        self.addReport("TFS", strategyReturns, signals, period)
        
        return strategyReturns
    
    def TF_simple_exec(prices, period):
        movingAverages = Strategy.getSimpleMovingAverageNp(prices, period)
        return TrendFollowing.signal(prices[-1], movingAverages[-1])
    
    def TF_exponential_bayes(self, alpha):
        valuesOpt = []
        for i in range(len(self.timeSeries.dataTrainList)):
            #self.timeSeries.setCurrentTrainData(i)
            self.timeSeries.setCurrentTrainDataNp(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_exponential(alpha)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def TF_exponential(self, alpha):
        prices = Strategy.getPricesNp(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        movingAverages = Strategy.getExponentialMovingAverageNp(prices, alpha)
        #moving_averages = Strategy.get_exponential_moving_average(prices, alpha)
        
        signals = np.zeros(np.shape(prices))
        for i in range(0, len(prices)):
            if startAfter > i:
                signals[i] = HOLD
                continue
            
            signals[i] = TrendFollowing.signal(prices[i], movingAverages[i])
            
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart, self.useSet, startAfter)
        if self.useSet == TEST:
            signals = signals[startAfter:] 
            
        self.addReport("TFE", strategyReturns, signals, alpha)
        
        return strategyReturns
    
    def TF_bb_rsi_bayes(self, bb_period, bb_std, rsi_period):
        valuesOpt = []
        for i in range(len(self.timeSeries.dataTrainList)):
            #self.timeSeries.setCurrentTrainData(i)
            self.timeSeries.setCurrentTrainDataNp(i)
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
        prices = Strategy.getPricesNp(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        bb_upper, bb_lower = Strategy.getBollingerBandsNp(prices, bb_period, bb_std)
        rsi = Strategy.getRsiNp(prices, rsi_period)
        
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
        if longPeriod <= shortPeriod:
            return 0
        
        for i in range(len(self.timeSeries.dataTrainList)):
            self.timeSeries.setCurrentTrainDataNp(i)
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
        prices = Strategy.getPricesNp(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        moving_averages_long = Strategy.getSimpleMovingAverageNp(prices, longPeriod)
        moving_averages_short = Strategy.getSimpleMovingAverageNp(prices, shortPeriod)
        
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
        if longAlpha >= shortAlpha:
            return 0
        
        for i in range(len(self.timeSeries.dataTrainList)):
            self.timeSeries.setCurrentTrainData(i)
            self.timeSeries.setCurrentTrainDataNp(i)
            self.setUseSet(TRAIN)
            strategyReturns = self.TF_crossover_exponential(longAlpha, shortAlpha)
            # returns = TimeSeries.pricesToReturns(strategyReturns)
            # sortino = statistics_1.sortino_ratio_annual(returns, "15MIN")
            # valuesOpt.append(sortino)
            valuesOpt.append(strategyReturns[-1])
                                    
        return sum(valuesOpt) / len(valuesOpt)
    
    def TF_crossover_exponential(self, longAlpha, shortAlpha):
            prices = Strategy.getPricesNp(self.timeSeries, self.useSet)
            startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
            moving_averages_long = Strategy.getExponentialMovingAverageNp(prices, longAlpha)
            moving_averages_short = Strategy.getExponentialMovingAverageNp(prices, shortAlpha)
            
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
