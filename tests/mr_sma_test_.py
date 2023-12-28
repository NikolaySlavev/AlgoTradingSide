from imports import *
from strategies.MeanReversion import MeanReversion
from timeSeries.BinanceTimeSeries import BinanceTimeSeries


if __name__ == "__main__":
    TRANSACTION_COST = 0.001
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    cash_start = 100
    numSplits = 4
    symbol = "BTCUSDT"
    days = 100
    interval = Client.KLINE_INTERVAL_1MINUTE
    
    
    untilThisDate = datetime.datetime(2023, 10, 10, 00, 00)
    sinceThisDate = untilThisDate - datetime.timedelta(days = days)
    warmupSinceThisDate = sinceThisDate - datetime.timedelta(days = days // numSplits)
    
    client = Client(config["BINANCE"]["bin_api_key"], config["BINANCE"]["bin_api_secret"])
    initialWarmupData = BinanceTimeSeries(client = client, config = config, dataPair = symbol, sinceThisDate = warmupSinceThisDate, untilThisDate = sinceThisDate, interval = interval, numSplits = 0, splitType = TimeSeriesSplitTypes.NONE, initialWarmupData = []).singleTimeSeries.dataFullNp
    timeSeries = BinanceTimeSeries(client = client, 
                                   config = config, 
                                   dataPair = symbol, 
                                   sinceThisDate = sinceThisDate, 
                                   untilThisDate = untilThisDate, 
                                   interval = interval, 
                                   numSplits = numSplits, 
                                   splitType = TimeSeriesSplitTypes.NORMAL,
                                   initialWarmupData = initialWarmupData)
    
    mr = MeanReversion(timeSeries, cash_start, len(initialWarmupData))
    max = len(initialWarmupData)
    
    pbounds = {'period': (2, max)}
    optimizerMRS = BayesianOptimization(f = mr.MR_simple_bayes, pbounds = pbounds, random_state = 1, verbose = 0, allow_duplicate_points = True)
    optimizerMRS.maximize(init_points = 20, n_iter = 50)
    mr.best_sma_period = int(optimizerMRS.max["params"]["period"])
    
    buy_hold_test_all = []
    mr_sma_test_all = []
    
    for iSplit in range(numSplits - 1):
        mr.timeSeries.setCurrentSingleTimeSeries(iSplit)
        mr.setUseSet("test")
        
        prices = mr.timeSeries.getPricesNp(useSet = "trainTest")
        prices_test = mr.timeSeries.getPricesNp(useSet = "test")
        returns = mr.timeSeries.getReturnsNp(useSet = "trainTest")
        returns_test = mr.timeSeries.getReturnsNp(useSet = "test")
        
        # BUY AND HOLD
        buy_hold_test_all.append(cash_start / prices_test[0] * prices_test)
        
        mr_sma_test_all.append(mr.MR_simple(mr.best_sma_period))
        

    assert mr.best_sma_period == 30722
    
    assert initialWarmupData[0][0] == 1.686096e+18
    assert initialWarmupData[-1][0] == 1.68825594e+18
    assert initialWarmupData[0][2] == 27230.07
    assert initialWarmupData[-1][2] == 30585.9
    
    assert prices[0] == 26188.65
    assert prices[-1] == 27584.91
    assert prices_test[0] == 26521.25
    assert prices_test[-1] == 27584.91
    
    assert buy_hold_test_all[0][-1] == 89.23183591420931
    assert buy_hold_test_all[1][-1] == 101.2756671306081
    assert buy_hold_test_all[2][-1] == 104.01059527737192
    
    assert mr_sma_test_all[0][-1] == 83.56294313183803
    assert mr_sma_test_all[1][-1] == 98.61674774129962
    assert mr_sma_test_all[2][-1] == 94.69562200019983
