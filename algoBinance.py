from binance.client import Client
from config import bin_api_key, bin_api_secret
from bayes_opt import BayesianOptimization
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import strategies
import statistics_1


if __name__ == "__main__":
    client = Client(bin_api_key, bin_api_secret)
    cash_start = 100
    numSplits = 10
    
    start_time = time.time()
    timeSeries = strategies.BinanceTimeSeries(client = client, dataPair = "BTCUSDT", howLong = 600, interval = Client.KLINE_INTERVAL_1HOUR, numSplits = numSplits)
    
    for i in range(len(timeSeries.dfTestList) - 1):
        timeSeries.dfTrainTest.append(timeSeries.dfTestList[i])
    
    # TREND FOLLOWING
    tf = strategies.TrendFollowing(timeSeries, cash_start)
 
    # MEAN REVERSION
    mr = strategies.MeanReversion(timeSeries, cash_start)

    # ARIMA
    arima_garch_model = strategies.ForecastArimaGarch(timeSeries, cash_start)
    
    max = len(timeSeries.dfTrain) * 0.9
    
    pbounds = {'period': (2 , max)}
    optimizerTFS = BayesianOptimization(f = tf.TF_simple_bayes, pbounds = pbounds, random_state = 1, verbose = 0, allow_duplicate_points = True)
    optimizerTFS.maximize(init_points = 15, n_iter = 100)
    best_tf_sma_period = int(optimizerTFS.max["params"]["period"])
     
    pbounds = {'alpha': (0.0001, 0.9999)}
    optimizerTFE = BayesianOptimization(f = tf.TF_exponential_bayes, pbounds = pbounds, random_state = 1, verbose = 0)
    optimizerTFE.maximize(init_points = 15, n_iter = 100)
    best_tf_ema_alpha = optimizerTFE.max["params"]["alpha"]
    
    pbounds = {'longPeriod': (2, max), "shortPeriod": (2, max)}
    optimizerTFCS = BayesianOptimization(f = tf.TF_crossover_simple_bayes, pbounds = pbounds, random_state = 1, verbose = 0, allow_duplicate_points = True)
    optimizerTFCS.maximize(init_points = 15, n_iter = 100)
    best_tf_cross_sma_long_period = int(optimizerTFCS.max["params"]["longPeriod"])
    best_tf_cross_sma_short_period = int(optimizerTFCS.max["params"]["shortPeriod"])
    
    pbounds = {'longAlpha': (0.0001, 0.9999), "shortAlpha": (0.0001, 0.9999)}
    optimizerTFCE = BayesianOptimization(f = tf.TF_crossover_exponential_bayes, pbounds = pbounds, random_state = 1, verbose = 0)
    optimizerTFCE.maximize(init_points = 15, n_iter = 100)
    best_tf_cross_ema_long_alpha = optimizerTFCE.max["params"]["longAlpha"]
    best_tf_cross_ema_short_alpha = optimizerTFCE.max["params"]["shortAlpha"]
    
    pbounds = {'bb_period': (2, max), "bb_std": (0.001, 5), "rsi_period": (2, max)}
    optimizerTFBR = BayesianOptimization(f = tf.TF_bb_rsi_bayes, pbounds = pbounds, random_state = 1, verbose = 0)
    optimizerTFBR.maximize(init_points = 20, n_iter = 120)
    best_tf_bb_period = int(optimizerTFBR.max["params"]["bb_period"])
    best_tf_bb_std = optimizerTFBR.max["params"]["bb_std"]
    best_tf_rsi_period = int(optimizerTFBR.max["params"]["rsi_period"])
    
    print("Best Trend Following SMA period is", optimizerTFS.max)
    print("Best Trend Following EMA alpha is", optimizerTFE.max)
    print("Best Trend Following Crossover SMA period is", optimizerTFCS.max)
    print("Best Trend Following Crossover EMA period is", optimizerTFCE.max)
    print("Best Trend Following BB period, BB std and RSI period are", optimizerTFBR.max)
            
    print("--- %s seconds ---\n" % (time.time() - start_time))
    
    buy_hold_test_all = []
    tf_sma_test_all = []
    tf_ema_test_all = []
    tf_cross_sma_test_all = []
    tf_cross_ema_test_all = []
    tf_bb_rsi_test_all = []
    
    for iSplit in range(numSplits):
        timeSeries.set_current_train_test_data(iSplit)
        tf.setUseSet("test")
        
        prices = timeSeries.get_prices(use_set = "trainTest")
        prices_test = timeSeries.get_prices(use_set = "test")
        returns = timeSeries.get_returns(use_set = "trainTest")
        returns_test = timeSeries.get_returns(use_set = "test")
        
        # BUY AND HOLD
        buy_hold_test_all.append(cash_start / prices_test[0] * prices_test)
        
        tf_sma_test_all.append(tf.TF_simple(best_tf_sma_period))
        tf_ema_test_all.append(tf.TF_exponential(best_tf_ema_alpha))
        tf_cross_sma_test_all.append(tf.TF_crossover_simple(best_tf_cross_sma_long_period, best_tf_cross_sma_short_period))
        tf_cross_ema_test_all.append(tf.TF_crossover_exponential(best_tf_cross_ema_long_alpha, best_tf_cross_ema_short_alpha))
        tf_bb_rsi_test_all.append(tf.TF_bb_rsi(best_tf_bb_period, best_tf_bb_std, best_tf_rsi_period))
    
    figure, axis = plt.subplots(nrows = numSplits, ncols = 1, figsize = (7, 7))
    for i in range(numSplits):
        strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_test_all[i]), 
                                                    ("TF SMA [" + str(best_tf_sma_period) + "]" , tf_sma_test_all[i]),
                                                    ("TF EMA [" + str(best_tf_ema_alpha) + "]" , tf_ema_test_all[i]),
                                                    ("TF CROSS SMA [" + str(best_tf_cross_sma_long_period) + ", " + str(best_tf_cross_sma_short_period) + "]" , tf_cross_sma_test_all[i]),
                                                    ("TF CROSS EMA [" + str(best_tf_cross_ema_long_alpha) + ", " + str(best_tf_cross_ema_short_alpha) + "]" , tf_cross_ema_test_all[i]), 
                                                    ("TF BB RSI [" + str(best_tf_bb_period) + "]", tf_bb_rsi_test_all[i])],
                                                    "Strategies Test Set", "Days", "Cash", axis = axis[i])
        axis[i].xaxis.set_major_locator(ticker.MaxNLocator(5))
        
        print("Buy Hold " + str(i) + " test is", buy_hold_test_all[i][-1])
        print(str("X") if buy_hold_test_all[i][-1] < tf_sma_test_all[i][-1] else str(""), "TF SMA " + str(i) + " test is", tf_sma_test_all[i][-1])
        print(str("X") if buy_hold_test_all[i][-1] < tf_ema_test_all[i][-1] else str(""), "TF EMA " + str(i) + " test is", tf_ema_test_all[i][-1])
        print(str("X") if buy_hold_test_all[i][-1] < tf_cross_sma_test_all[i][-1] else str(""), "TF CS " + str(i) + " test is", tf_cross_sma_test_all[i][-1])
        print(str("X") if buy_hold_test_all[i][-1] < tf_cross_ema_test_all[i][-1] else str(""), "TF CE " + str(i) + " test is", tf_cross_ema_test_all[i][-1])
        print(str("X") if buy_hold_test_all[i][-1] < tf_bb_rsi_test_all[i][-1] else str(""), "TF BR " + str(i) + " test is", tf_bb_rsi_test_all[i][-1])
        print("")
            
    plt.show()
    exit()
    
    for iSplit in range(numSplits):
        timeSeries.set_current_train_test_data(iSplit)
        tf.setUseSet("test")
        
        prices = timeSeries.get_prices(use_set = "trainTest")
        prices_test = timeSeries.get_prices(use_set = "test")
        returns = timeSeries.get_returns(use_set = "trainTest")
        returns_test = timeSeries.get_returns(use_set = "test")
        
        # BUY AND HOLD
        buy_hold_test = cash_start / prices_test[0] * prices_test
        
        tf_sma_test = tf.TF_simple(best_tf_sma_period)
        tf_ema_test = tf.TF_exponential(best_tf_ema_alpha)
        tf_cross_sma_test = tf.TF_crossover_simple(best_tf_cross_sma_long_period, best_tf_cross_sma_short_period)
        tf_cross_ema_test = tf.TF_crossover_exponential(best_tf_cross_ema_long_alpha, best_tf_cross_ema_short_alpha)
        tf_bb_rsi_test = tf.TF_bb_rsi(best_tf_bb_period, best_tf_bb_std, best_tf_rsi_period)

        #timeSeries.reports.writeReport("iter" + str(iSplit) + "/")
        #timeSeries.reports.cleanVars()

        figure, axis = plt.subplots(nrows = 2, ncols = 2, figsize=(7, 7))
        axis[0, 0].plot(prices) 
        axis[0, 0].plot(prices_test, color = "orange")
        axis[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(5))
        axis[0, 0].set_title("Prices")
        axis[0, 1].plot(returns)
        axis[0, 1].plot(returns_test, color = "orange")
        axis[0, 1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        axis[0, 1].set_title("Returns")
        
        strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_test), 
                                                    ("TF SMA [" + str(best_tf_sma_period) + "]" , tf_sma_test), 
                                                    ("TF EMA [" + str(best_tf_ema_alpha) + "]" , tf_ema_test), 
                                                    ("TF CROSS SMA [" + str(best_tf_cross_sma_long_period) + ", " + str(best_tf_cross_sma_short_period) + "]" , tf_cross_sma_test),
                                                    ("TF CROSS EMA [" + str(best_tf_cross_ema_long_alpha) + ", " + str(best_tf_cross_ema_short_alpha) + "]" , tf_cross_ema_test), 
                                                    ("TF BB RSI [" + str(best_tf_bb_period) + "]", tf_bb_rsi_test)],
                                                    "Strategies Test Set", "Days", "Cash", axis = axis[1, 1])
        axis[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        
        plt.show()    
    
    exit()

    for iSplit in range(numSplits):
        timeSeries.set_current_train_test_data(iSplit)
        
        prices = timeSeries.get_prices(use_set = "trainTest")
        prices_train = timeSeries.get_prices(use_set = "train")
        prices_test = timeSeries.get_prices(use_set = "test")
        returns = timeSeries.get_returns(use_set = "trainTest")
        returns_train = timeSeries.get_returns(use_set = "train")
        returns_test = timeSeries.get_returns(use_set = "test")
        
        # TREND FOLLOWING
        tf = strategies.TrendFollowing(timeSeries, cash_start)

        # MEAN REVERSION
        mr = strategies.MeanReversion(timeSeries, cash_start)

        # ARIMA
        arima_garch_model = strategies.ForecastArimaGarch(timeSeries, cash_start)
        
        # BUY AND HOLD
        buy_hold = cash_start / prices[0] * prices
        buy_hold_train = buy_hold[:len(returns_train)]
        buy_hold_test = cash_start / prices_test[0] * prices_test


        
        # STRATEGIES PARAMETER TUNING
        sma_periods = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 150, 200, 24*20, 24*30, 24*35, 24*40, 24*45, 24*50]
        ema_alphas = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        bb_periods = [5, 10, 20, 30, 40, 50, 60, 80]#, 100]#, 150, 200]
        bb_stds = [0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3]
        rsi_periods = [5, 10, 15, 20, 30]


        # best_sma_tf_periods = 0
        # best_sma_tf = 0
        # tf_sma_train_best = None
        # for i in range(len(sma_periods)):
        #     tf_sma_train = tf.TF_all_old(ma_type = "simple", parameter = sma_periods[i], use_set = "train")
        #     if tf_sma_train[-1] > best_sma_tf and tf_sma_train[-1] != cash_start:
        #         best_sma_tf = tf_sma_train[-1]
        #         best_sma_tf_periods = sma_periods[i]
        #         tf_sma_train_best = tf_sma_train

        # print("Best Trend Following SMA period is", best_sma_tf_periods)

        # best_ema = 0
        # best_ema_tf = 0
        # tf_ema_train_best = None
        # for i in range(len(ema_alphas)):
        #     tf_ema_train = tf.TF_all(ma_type = "exponential", parameter = ema_alphas[i], use_set = "train")
        #     if tf_ema_train[-1] > best_ema_tf and tf_ema_train[-1] != cash_start:
        #         best_ema_tf = tf_ema_train[-1]
        #         best_ema = ema_alphas[i]
        #         tf_ema_train_best = tf_ema_train

        # print("Best Trend Following EMA alpha is", best_ema)
                
        # best_tf_cross_sma_periods = 0
        # best_cross_sma_tf = 0
        # tf_cross_sma_train_best = None
        # for i in range(len(sma_periods) - 1):
        #     for j in range(i + 1, len(sma_periods)):
        #         tf_cross_sma_train = tf.TF_all(ma_type = "crossover_simple", parameter = {"long_period": sma_periods[j], "short_period": sma_periods[i]}, use_set = "train")
        #         if tf_cross_sma_train[-1] > best_cross_sma_tf and tf_cross_sma_train[-1] != cash_start:
        #             best_cross_sma_tf = tf_cross_sma_train[-1]
        #             best_tf_cross_sma_periods = {"long_period": sma_periods[j], "short_period": sma_periods[i]}
        #             tf_cross_sma_train_best = tf_cross_sma_train

        # print("Best Trend Following Crossover SMA period is", best_tf_cross_sma_periods)
                
        # best_tf_bb_rsi_params = [0,0,0]
        # best_tf_bb_rsi = 0
        # tf_bb_rsi_train_best = None
        # for i in range(len(bb_periods)):
        #     for j in range(len(bb_stds)):
        #         for k in range(len(rsi_periods)):        
        #             tf_bb_rsi_train = tf.TF_all(ma_type = "bb_rsi", parameter = {"bb_period": bb_periods[i], "bb_std": bb_stds[j], "rsi_period": rsi_periods[k]}, use_set = "train")
        #             # restrictive if condition to ensure that if the strategy is downward it won't train it to do nothing and will accept the losing one + it will try to find a strategy where at least 20% of the time there has been movement
        #             if tf_bb_rsi_train[-1] > best_tf_bb_rsi and tf_bb_rsi_train[-1] != cash_start and tf_bb_rsi_train.tolist().count(cash_start) < len(tf_bb_rsi_train) * 0.8:
        #                 best_tf_bb_rsi = tf_bb_rsi_train[-1]
        #                 best_tf_bb_rsi_params = {"bb_period": bb_periods[i], "bb_std": bb_stds[j], "rsi_period": rsi_periods[k]}
        #                 tf_bb_rsi_train_best = tf_bb_rsi_train

        #print("Best Trend Following BB period, BB std and RSI period are", best_tf_bb_rsi_params)
            
        # best_sma_mr_periods = 0
        # best_sma_mr = 0
        # mr_sma_train_best = None
        # for i in range(len(sma_periods)):
        #     mr_sma_train = mr.MR_all(ma_type = "simple", parameter = sma_periods[i], use_set = "train")
        #     if mr_sma_train[-1] > best_sma_mr and mr_sma_train[-1] != cash_start:
        #         best_sma_mr = mr_sma_train[-1]
        #         best_sma_mr_periods = sma_periods[i]
        #         mr_sma_train_best = mr_sma_train

        # print("Best Mean Reversion SMA period is", best_sma_mr_periods)
        
        # best_mr_params = [0,0,0]
        # best_mr = 0
        # mr_bb_rsi_train_best = None
        # for i in range(len(bb_periods)):
        #     for j in range(len(bb_stds)):
        #         for k in range(len(rsi_periods)):        
        #             mr_bb_rsi_train = mr.MR_all(ma_type = "bb_rsi", parameter = {"bb_period": bb_periods[i], "bb_std": bb_stds[j], "rsi_period": rsi_periods[k]}, use_set = "train")
        #             if mr_bb_rsi_train[-1] > best_mr and mr_bb_rsi_train[-1] != cash_start and mr_bb_rsi_train.tolist().count(cash_start) < len(mr_bb_rsi_train)*0.8:
        #                 best_mr = mr_bb_rsi_train[-1]
        #                 best_mr_params = {"bb_period": bb_periods[i], "bb_std": bb_stds[j], "rsi_period": rsi_periods[k]}
        #                 mr_bb_rsi_train_best = mr_bb_rsi_train

        # print("Best Mean Reversion BB period, BB std and RSI period are", best_mr_params)

        #arima_train, arima_pred_train, arima_order_train = arima_garch_model.ARIMA_GARCH_train()

        tf_sma_test = tf.TF_all(ma_type = "simple", parameter = best_sma_tf_periods, use_set = "test")
        tf_ema_test = tf.TF_all(ma_type = "exponential", parameter = best_ema, use_set = "test")
        tf_cross_sma_test = tf.TF_all(ma_type = "crossover_simple", parameter = best_tf_cross_sma_periods, use_set = "test")
        tf_bb_rsi_test = tf.TF_all(ma_type = "bb_rsi", parameter = best_mr_params, use_set = "test")
        mr_sma_test = mr.MR_all(ma_type = "simple", parameter = best_sma_mr_periods, use_set = "test")
        mr_bb_rsi_test = mr.MR_all(ma_type = "bb_rsi", parameter = best_mr_params, use_set = "test")
        #arima_test, arima_pred_test, arima_buy_sell_test, arima_order_test = arima_garch_model.ARIMA_GARCH_test()

        timeSeries.reports.writeReport("iter" + str(iSplit) + "/")
        timeSeries.reports.cleanVars()

        figure, axis = plt.subplots(nrows = 2, ncols = 2, figsize=(7, 7))
        axis[0, 0].plot(prices) 
        axis[0, 0].plot(prices_test, color = "orange")
        axis[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(5))
        axis[0, 0].set_title("Prices")
        axis[0, 1].plot(returns)
        axis[0, 1].plot(returns_test, color = "orange")
        axis[0, 1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        axis[0, 1].set_title("Returns")
        
        strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_train), 
                                                    ("TF SMA [" + str(best_sma_tf_periods) + "]", tf_sma_train_best),
                                                    ("TF EMA [" + str(best_ema) + "]", tf_ema_train_best), 
                                                    ("TF CROSS SMA [" + str(best_tf_cross_sma_periods) + "]", tf_cross_sma_train_best), 
                                                    ("TF BB RSI [" + str(best_tf_bb_rsi_params) + "]", tf_bb_rsi_train_best),
                                                    ("MR SMA [" + str(best_sma_mr_periods) + "]", mr_sma_train),
                                                    ("MR BB RSI [" + str(best_mr_params) + "]", mr_bb_rsi_train_best)],
                                                    #("ARIMA+GARCH" + str(arima_order_train), arima_train)], 
                                                    "Strategies Train Set", "Days", "Cash", axis = axis[1, 0])
        axis[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(5))
        
        strategies.SyntheticTimeSeries.plot([("Buy and Hold", buy_hold_test), 
                                                    ("TF SMA [" + str(best_sma_tf_periods) + "]" , tf_sma_test), 
                                                    ("TF EMA [" + str(best_ema) + "]" , tf_ema_test), 
                                                    ("TF CROSS SMA [" + str(best_tf_cross_sma_periods) + "]" , tf_cross_sma_test), 
                                                    ("TF BB RSI [" + str(best_tf_bb_rsi_params) + "]", tf_bb_rsi_test),
                                                    ("MR SMA [" + str(best_sma_mr_periods) + "]", mr_sma_test),
                                                    ("MR BB RSI [" + str(best_mr_params) + "]", mr_bb_rsi_test)],
                                                    #("ARIMA+GARCH" + str(arima_order_test), arima_test)], 
                                                    "Strategies Test Set", "Days", "Cash", axis = axis[1, 1])
        axis[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        
        plt.show()
