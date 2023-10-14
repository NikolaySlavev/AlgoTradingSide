from imports import *
from strategies.Strategy import Strategy
from timeSeries.TimeSeries import TimeSeries


"""
    Class that encapsulates the Mean Reversion Strategy
    Implemented both SMA and Bands+RSI
"""
class MeanReversion(Strategy):
    def __init__(self, timeSeries, cashStart, warmupSize):
        if not isinstance(timeSeries, TimeSeries):
            return "Object needs to be an instance of SyntheticTimeSeries"

        super(MeanReversion, self).__init__()
        self.timeSeries = timeSeries
        self.cashStart = cashStart
        self.warmupSize = warmupSize
    
    def signal(price, ma):
        if ma > price: 
            return BUY
        else: 
            return SELL
        
    @numba.jit((numba.float64[:], numba.float64[:]), nopython = True, nogil = True)
    def getSignals(prices, mas):
        signals = np.zeros(np.shape(prices))
        for i in range(len(prices)):
            if mas[i] > prices[i]:
                signals[i] = BUY
            else:
                signals[i] =  SELL
                
        return signals
    
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
        for i in range(len(self.timeSeries.singleTimeSeriesList)):
            self.timeSeries.setCurrentSingleTimeSeries(i)
            self.setUseSet(TRAIN)
            valuesOpt.append(self.MR_simple(period)[-1])
                        
        return sum(valuesOpt) / len(valuesOpt)
    
    def MR_exponential_bayes(self, alpha):
        valuesOpt = []
        for i in range(len(self.timeSeries.dfTrainList)):
            self.timeSeries.setCurrentTrainDataNp(i)
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
        pricesForMa = self.timeSeries.getPricesNp(self.useSet + WARMUP)
        prices = self.timeSeries.getPricesNp(self.useSet)
        startAfter = self.timeSeries.getWarmupSize(self.useSet)
        movingAverages = Strategy.getSimpleMovingAverageNp(pricesForMa, period)
        movingAverages = movingAverages[startAfter:]
        signals = MeanReversion.getSignals(prices, movingAverages)
        strategyReturns = Strategy.getStrategyReturns(prices, signals, self.cashStart)
                
        self.addReport("MRS", strategyReturns, signals, period)
        return strategyReturns
    
    def MR_exponential_exec(prices, alpha):
        movingAverages = Strategy.getExponentialMovingAverageNp(prices, alpha)
        return MeanReversion.signal(prices[-1], movingAverages[-1])
    
    def MR_exponential(self, alpha):
        prices = Strategy.getPricesNp(self.timeSeries, self.useSet)
        startAfter = Strategy.getStartAfter(self.timeSeries, self.useSet)
        movingAverages = Strategy.getExponentialMovingAverageNp(prices, alpha)
        signals = MeanReversion.getSignals(prices, movingAverages)
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
