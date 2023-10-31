from imports import *
from strategies.MeanReversion import MeanReversion
from timeSeries.TimeSeries import TimeSeries
from timeSeries.BinanceTimeSeries import BinanceTimeSeries
from statistics_1 import *
from reporting.Reporting import writeReportCsv

#import cProfile

if __name__ == "__main__":
#def main():
    start_time = time.time()
    
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    
    cash_start = 100
    numSplits = 4
    symbol = "BTCUSDT"
    days = 100
    interval = Client.KLINE_INTERVAL_1MINUTE
    
    untilThisDate = datetime.datetime.now(datetime.timezone.utc)
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
    
    print("--- %s seconds ---\n" % (time.time() - start_time))
    
    # for i in range(len(timeSeries.dataTestList) - 1):
    #     timeSeries.dataTrainList.append(timeSeries.dataTestList[i])
    #     timeSeries.dataTestList.append([])
    #     timeSeries.dataTrainTestList.append(timeSeries.dataTestList[i])
    
    
    # MEAN REVERSION
    mr = MeanReversion(timeSeries, cash_start, len(initialWarmupData))
    
    max = len(initialWarmupData)
    
    #mr.timeSeries.plotPrices(useSet = "trainTest")
        
    #pd.DataFrame(mr.timeSeries.dataFullNp, columns = list(mr.timeSeries.columnsNp.keys())).to_csv(r"plotData\plotPrices.csv", index = False)
    
    ############### MEAN REVERSION ########################    
    pbounds = {'period': (2, max)}
    optimizerMRS = BayesianOptimization(f = mr.MR_simple_bayes, pbounds = pbounds, random_state = 1, verbose = 0, allow_duplicate_points = True)
    optimizerMRS.maximize(init_points = 20, n_iter = 50)
    mr.best_sma_period = int(optimizerMRS.max["params"]["period"])
        
    # pbounds = {'alpha': (0.0001, 1)}
    # optimizerMRE = BayesianOptimization(f = mr.MR_exponential_bayes, pbounds = pbounds, random_state = 1, verbose = 0)
    # optimizerMRE.maximize(init_points = 15, n_iter = 100)
    # best_mr_ema_alpha = optimizerMRE.max["params"]["alpha"]
    
    # pbounds = {'longPeriod': (2, max), "shortPeriod": (2, max)}
    # optimizerMRCS = BayesianOptimization(f = mr.MR_crossover_simple_bayes, pbounds = pbounds, random_state = 1, verbose = 0)
    # optimizerMRCS.maximize(init_points = 15, n_iter = 100)
    # best_mr_cross_sma_long_period = int(optimizerMRCS.max["params"]["longPeriod"])
    # best_mr_cross_sma_short_period = int(optimizerMRCS.max["params"]["shortPeriod"])
    
    # pbounds = {'longAlpha': (0.0001, 1), "shortAlpha": (0.0001, 1)}
    # optimizerMRCE = BayesianOptimization(f = mr.MR_crossover_exponential_bayes, pbounds = pbounds, random_state = 1, verbose = 0)
    # optimizerMRCE.maximize(init_points = 15, n_iter = 100)
    # best_mr_cross_ema_long_alpha = optimizerMRCE.max["params"]["longAlpha"]
    # best_mr_cross_ema_short_alpha = optimizerMRCE.max["params"]["shortAlpha"]
        
    # pbounds = {'bb_period': (2, max), "bb_std": (0.001, 5), "rsi_period": (2, max)}
    # optimizerMRBR = BayesianOptimization(f = mr.MR_bb_rsi_bayes, pbounds = pbounds, random_state = 1, verbose = 0)
    # optimizerMRBR.maximize(init_points = 15, n_iter = 100)
    # best_mr_bb_period = int(optimizerMRBR.max["params"]["bb_period"])
    # best_mr_bb_std = optimizerMRBR.max["params"]["bb_std"]
    # best_mr_rsi_period = int(optimizerMRBR.max["params"]["rsi_period"])
    
    print("Best Mean Reversion SMA period is", optimizerMRS.max)
    # print("Best Msean Reversion EMA alpha is", optimizerMRE.max)
    # print("Best Mean Reversion Crossover SMA period is", optimizerMRCS.max)
    # print("Best Mean Reversion Crossover EMA period is", optimizerMRCE.max)
    # print("Best Mean Reversion BB period, BB std and RSI period are", optimizerMRBR.max)
    
    print("--- %s seconds ---\n" % (time.time() - start_time))
    
    writeReportCsv(mr)
    
    buy_hold_test_all = []
    mr_sma_test_all = []
    mr_ema_test_all = []
    mr_cross_sma_test_all = []
    mr_cross_ema_test_all = []
    mr_bb_rsi_test_all = []
    
    for iSplit in range(numSplits - 1):
        mr.timeSeries.setCurrentSingleTimeSeries(iSplit)
        mr.setUseSet("test")
        
        prices = mr.timeSeries.getPricesNp(useSet = "trainTest")
        prices_test = mr.timeSeries.getPricesNp(useSet = "test")
        returns = mr.timeSeries.getReturnsNp(useSet = "trainTest")
        returns_test = mr.timeSeries.getReturnsNp(useSet = "test")
        
        # BUY AND HOLD
        buy_hold_test_all.append(cash_start / prices_test[0] * prices_test)
        
        mr_sma_test_all.append(mr.MR_simple(mr.best_mr_sma_period))
        # mr_ema_test_all.append(mr.MR_exponential(best_mr_ema_alpha))
        # mr_cross_sma_test_all.append(mr.MR_crossover_simple(best_mr_cross_sma_long_period, best_mr_cross_sma_short_period))
        # mr_cross_ema_test_all.append(mr.MR_crossover_exponential(best_mr_cross_ema_long_alpha, best_mr_cross_ema_short_alpha))
        # mr_bb_rsi_test_all.append(mr.MR_bb_rsi(best_mr_bb_period, best_mr_bb_std, best_mr_rsi_period))
    
    #mr.timeSeries.plotIndicators(useSet = "trainTest")
    
    figure, axis = plt.subplots(nrows = numSplits - 1, ncols = 1, figsize = (7, 7))
    for i in range(numSplits - 1):
        TimeSeries.plot([("Buy and Hold", buy_hold_test_all[i]), 
                                                    ("MR SMA [" + str(mr.best_mr_sma_period) + "]" , mr_sma_test_all[i])],
                                                    # ("MR EMA [" + str(best_mr_ema_alpha) + "]" , mr_ema_test_all[i]),
                                                    # ("MR CROSS SMA [" + str(best_mr_cross_sma_long_period) + ", " + str(best_mr_cross_sma_short_period) + "]" , mr_cross_sma_test_all[i]),
                                                    # ("MR CROSS EMA [" + str(best_mr_cross_ema_long_alpha) + ", " + str(best_mr_cross_ema_short_alpha) + "]" , mr_cross_ema_test_all[i]),
                                                    # ("MR BB RSI [" + str(best_mr_bb_period) + "]", mr_bb_rsi_test_all[i])],
                                                    "Strategies Test Set", "Days", "Cash", axis = axis[i], xValues = pd.to_datetime(mr.timeSeries.getDates(useSet = "test")))
        axis[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        
        print("Buy Hold " + str(i) + " test is", buy_hold_test_all[i][-1])
        print(str("X") if buy_hold_test_all[i][-1] < mr_sma_test_all[i][-1] else str(""), "MR SMA " + str(i) + " test is", mr_sma_test_all[i][-1])
        # print(str("X") if buy_hold_test_all[i][-1] < mr_ema_test_all[i][-1] else str(""), "MR EMA " + str(i) + " test is", mr_ema_test_all[i][-1])
        # print(str("X") if buy_hold_test_all[i][-1] < mr_cross_sma_test_all[i][-1] else str(""), "MR CS " + str(i) + " test is", mr_cross_sma_test_all[i][-1])
        # print(str("X") if buy_hold_test_all[i][-1] < mr_cross_ema_test_all[i][-1] else str(""), "MR CE " + str(i) + " test is", mr_cross_ema_test_all[i][-1])
        # print(str("X") if buy_hold_test_all[i][-1] < mr_bb_rsi_test_all[i][-1] else str(""), "MR BR " + str(i) + " test is", mr_bb_rsi_test_all[i][-1])
        print("")
            
    plt.show()
    
# if __name__ == "__main__":
#     cProfile.run('main()')
#     exit()