from imports import *
from strategies.Strategy import Strategy
from timeSeries.TimeSeries import TimeSeries


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
        self.pricesTrain = self.timeSeries.get_prices(use_set = TRAIN)
        self.pricesTest = self.timeSeries.get_prices(use_set = TEST)
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
            
        auto_arima_model = pm.auto_arima(self.pricesTrain, start_p = 1, start_q = 1, max_p = 20, max_q = 20, trace = False)
        #auto_arima_model = pm.auto_arima(self.returnsTrain, start_p = 1, start_q = 1, max_p = 20, max_q = 20, trace = False)
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
        arima_strategy = pd.Series(arima_strategy, index = self.pricesTest.index)
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
        arima_prediction = np.zeros(np.shape(self.returnsTrain)) # pd.Series(0, index = self.returnsTrain.index)
        w = np.zeros(np.shape(self.returnsTrain))
        cash = np.zeros(np.shape(self.returnsTrain))
        cash[0] = self.cash_start
        
        auto_arima_model = pm.auto_arima(self.pricesTrain, start_p = 1, start_q = 1, max_p = 20, max_q = 20, trace = False)
        arima_prediction = auto_arima_model.predict_in_sample()
        for i in range(len(arima_prediction) - 1):
            cash[i+1], w[i+1], _ = ForecastArimaGarch.ARIMA_position(arima_prediction[i], cash[i], w[i], self.pricesTrain[i])
                
        arima_strategy = [a*b for a,b in zip(w, self.pricesTrain)] + cash
        arima_strategy = pd.Series(arima_strategy, index = self.pricesTrain.index)
        return arima_strategy, arima_prediction, str(auto_arima_model.order)
