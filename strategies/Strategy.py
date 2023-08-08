# https://proxy-seller.com/personal/orders/
from imports import *


TESTR = None

class Strategy():
    def __init__(self):
        self.useSet = TRAIN
        self.timeSeries = None
    
    # def rollingWindow(prices, period):
    #     pad = np.ones(len(prices.shape), dtype = np.int32)
    #     pad[-1] = period - 1
    #     pad = list(zip(pad, np.zeros(len(prices.shape), dtype = np.int32)))
    #     prices = np.pad(prices, pad, mode = 'reflect')
    #     shape = prices.shape[:-1] + (prices.shape[-1] - period + 1, period)
    #     strides = prices.strides + (prices.strides[-1],)
    #     return np.lib.stride_tricks.as_strided(prices, shape = shape, strides = strides)
    
    @numba.jit((numba.float64[:], numba.int64), nopython = True, nogil = True)
    def rollingWindow(prices, period):
        shape = prices.shape[:-1] + (prices.shape[-1] - period + 1, period)
        strides = prices.strides + (prices.strides[-1],)
        return np.lib.stride_tricks.as_strided(prices, shape = shape, strides = strides)
    
    def test111(prices, period):
        return [prices[i] for i in range(period - 1)]
    
    def test222(prepend, ma):
        return np.concatenate([prepend, ma])
    
    #@numba.jit((numba.float64[:, :]), nopython = True, nogil = True)
#    @numba.jit(numba.types.UniTuple(numba.float64[:,:], 1), nopython = True, nogil = True)
    #@numba.jit((numba.typeof(((1.2, 1.3, 1.4), (2.1, 3.2, 4.1)))), nopython = True, nogil = True)
    #@numba.jit((numba.float64[:, :]), nopython = True, nogil = True)
    def test3332(rollingPrices):
        res = []
        for prices in rollingPrices:
            sum = 0
            for p in prices:
                sum += p
                
            r = sum / len(prices)
            res.append(r)
            
        return res
    
    #@numba.jit((numba.float64[:]), nopython = True, nogil = True)
    def test333(test):
        return np.mean(test, axis = 1)
    
    @numba.guvectorize(['void(float64[:], intp[:], float64[:])'], '(n),()->(n)')
    def getSimpleMovingAverageNumba(a, window_arr, out):
        window_width = window_arr[0]
        asum = 0.0
        count = 0
        for i in range(window_width):
            asum += a[i]
            count += 1
            out[i] = asum / count
        for i in range(window_width, len(a)):
            asum += a[i] - a[i - window_width]
            out[i] = asum / count
    
    #def getSimpleMovingAverageNp(prices, period):
        # need to add padding in the beginning
        #prepend = Strategy.test111(prices, period)
        #test = Strategy.rollingWindow(prices, period)
        #ma = Strategy.test333(test)
        #ma2 = Strategy.move_mean(prices, period)
        #ma = Strategy.test3332(test)
        #return Strategy.test222(prepend, ma)
    
    def getSimpleMovingAverageNp(prices, period):
        return Strategy.getSimpleMovingAverageNumba(prices, period)
    
    def getRollingStdNp(prices, period):
        prepend = [0 for _ in range(period - 1)]
        std = np.std(Strategy.rollingWindow(prices, period), axis = 1)
        return np.concatenate([prepend, std])
    
    # Computes SMA
    def get_simple_moving_average(prices, period):
        ma = prices.rolling(window = period).mean()
        
        # to avoid NA values in the first entries
        for i in range(period - 1):
            ma[i] = prices[i]
        
        return ma
    
    @numba.jit((numba.float64[:], numba.float64), nopython=True, nogil=True)
    def _ewma(arr_in, alpha):
        r"""Exponentialy weighted moving average specified by a decay window
        to provide better adjustments for small windows via
        """
        n = arr_in.shape[0]
        ewma = np.empty(n, dtype = np.float64)
        w = 1
        ewma_old = arr_in[0]
        ewma[0] = ewma_old
        for i in range(1, n):
            w += (1-alpha)**i
            ewma_old = ewma_old*(1-alpha) + arr_in[i]
            ewma[i] = ewma_old / w

        return ewma
    
    @numba.jit((numba.float64[:], numba.float64), nopython = True, nogil = True)
    def getExponentialMovingAverageNp(prices, alpha):
        n = prices.shape[0]
        ewma = np.empty(n, dtype = np.float64)
        ewma[0] = prices[0]
        for i in range(1, n):
            ewma[i] = prices[i] * alpha + ewma[i-1] * (1 - alpha)
            
        return ewma
    
    # Computes EMA
    def get_exponential_moving_average(prices, alpha):
        return prices.ewm(alpha = alpha, adjust = False).mean()
    
    def getBollingerBandsNp(prices, period = 20, numStd = 2):
        ma = Strategy.getSimpleMovingAverageNp(prices, period)
        std = Strategy.getRollingStdNp(prices, period)
        upper = ma + numStd * std
        lower = ma - numStd * std
        return upper, lower
    
    # Compute the Bands
    def get_bollinger_bands(prices, period = 20, num_std = 2):
        ma = Strategy.get_simple_moving_average(prices, period)
        std = prices.rolling(period).std() # don't do rolling twice
        upper = ma + num_std * std
        lower = ma - num_std * std
        for i in range(period):
            upper[i] = prices[i]
            lower[i] = prices[i]
           
        return upper, lower

    def getRsiNp(prices, period):
        delta = np.diff(prices) #, prepend = prices[0])
        gain2 = np.clip(delta, a_min = 0, a_max = None)
        loss2 = -1 * np.clip(delta, a_min = None, a_max = 0)
        sma_gain = np.mean(Strategy.rollingWindow(gain2, period), axis = 1)
        sma_loss = np.mean(Strategy.rollingWindow(loss2, period), axis = 1)
        rs = sma_gain / sma_loss
        rsi = 100 - (100 / (rs + 1))
        
        # period - 1 for the rolling and + 1 for the diff
        prepend = [50 for _ in range(period)]
        rsi = np.concatenate([prepend, rsi])        
        return rsi

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
        #if signal == BUY and last_signal != BUY or signal == SELL and last_signal != SELL:
        # NOTE: won't work with HOLD
        if signal != last_signal:
            return TRANSACTION_COST
        else:
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
    
    @numba.jit((numba.float64, numba.int64, numba.float64, numba.float64, numba.float64), nopython = True, nogil = True)
    def position(price, signal, last_signal, prev_cash, prev_w):
        #transactionCost = Strategy.get_transaction_cost(signal, last_signal)
        if signal != last_signal:
            transactionCost = TRANSACTION_COST
        else:
            transactionCost = 0
            
        if signal == BUY:
            # BUY
            w = (prev_cash * (1 - transactionCost)) / price  + prev_w
            cash = 0
        else:
            # SELL
            cash = prev_w * (1 - transactionCost) * price + prev_cash
            w = 0
            
        return cash, w    
    
    @numba.jit((numba.float64[:], numba.float64[:], numba.float64), nopython = True, nogil = True)
    def positions(prices, signals, cash_start):
        w = np.zeros(np.shape(prices))
        cash = np.zeros(np.shape(prices))
        
        if signals[0] != HOLD:
            transactionCost = TRANSACTION_COST
        else:
            transactionCost = 0
            
        if signals[0] == BUY:
            # BUY
            w[0] = (cash_start * (1 - transactionCost)) / prices[0]  + 0
            cash[0] = 0
        else:
            # SELL
            cash[0] = 0 * (1 - transactionCost) * prices[0] + cash_start
            w[0] = 0
        
        for i in range(1, len(prices)):
            if signals[i] != signals[i-1]:
                transactionCost = TRANSACTION_COST
            else:
                transactionCost = 0
            
            if signals[i] == BUY:
                # BUY
                w[i] = (cash[i-1] * (1 - transactionCost)) / prices[i]  + w[i-1]
                cash[i] = 0
            else:
                # SELL
                cash[i] = w[i-1] * (1 - transactionCost) * prices[i] + cash[i-1]
                w[i] = 0
            # else:
            #     cash[i] = cash[i-1]
            #     w[i] = w[i-1]
                
        return cash, w
    
    
    
    @numba.jit((numba.float64[:], numba.float64[:], numba.float64[:]), nopython = True, nogil = True)
    def computeReturns(w, prices, cash):
        returns = np.zeros(np.shape(prices))
        for i in range(len(prices)):
            returns[i] = w[i] * prices[i] + cash[i]
            
        return returns
        #return [a * b for a, b in zip(w, prices)] + cash
    
    def getStrategyReturns(prices, signals, cashStart, useSet, startAfter):
        #w = np.zeros(np.shape(prices))
        #cash = np.zeros(np.shape(prices))
        #lastSignal = HOLD
        #cash[0], w[0] = Strategy.position(prices[0], signals[0], lastSignal, cashStart, 0)        
        cash, w = Strategy.positions(prices, signals, cashStart)
        strategyReturns = Strategy.computeReturns(w, prices, cash)
        if useSet == TEST:
            strategyReturns = strategyReturns[startAfter:]
        
        return strategyReturns
    
    def setUseSet(self, useSet):
        self.useSet = useSet
    
    def getPricesNp(timeSeries, useSet):
        if useSet == TEST:
            return timeSeries.getPricesNp(useSet = TRAINTEST)

        return timeSeries.getPricesNp(useSet = useSet)
    
    def getPrices(timeSeries, useSet):
        if useSet == TEST:
            return timeSeries.getPrices(useSet = TRAINTEST)

        return timeSeries.getPrices(useSet = useSet)
        
    def getStartAfter(timeSeries, useSet):
        return len(timeSeries.dataTrainNp) if useSet == TEST else 0
    
    def addReport(self, strategyName, strategyReturns, signals, params):
        if not self.timeSeries.reportEnabled:
            return
        
        df = self.timeSeries.get_set(use_set = self.useSet).copy()
        df["strategyReturns"] = strategyReturns
        df["signals"] = signals
        self.timeSeries.reports.addReport(strategyName = strategyName, params = params, finalCash = strategyReturns[-1], df = df)
    