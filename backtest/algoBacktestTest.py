from imports import *
from strategies.MeanReversion import TrendFollowing
from timeSeries.BinanceTimeSeries import BinanceTimeSeries
from timeSeries.TimeSeries import TimeSeries


if __name__ == "__main__":
    cash_start = 10
        
    client = Client(bin_api_key, bin_api_secret)
    timeSeries = BinanceTimeSeries(client = client, dataPair = "BTCUSDT", howLong = 900, interval = Client.KLINE_INTERVAL_1HOUR, numSplits = 2)
    timeSeries.set_current_train_test_data(1)
    timeSeries.reportEnabled = True
    
    # TREND FOLLOWING
    tf = TrendFollowing(timeSeries, cash_start)
    best_tf_sma_period = 570
    best_tf_ema_alpha = 0.04301
    best_tf_cross_sma_long_period = 898.94
    best_tf_cross_sma_short_period = 3.945
    best_tf_cross_ema_long_alpha = 0.783906
    best_tf_cross_ema_short_alpha = 0.660514
    best_tf_bb_period = 511.759
    best_tf_bb_std = 0.0093612
    best_tf_rsi_period = 5.53644825
    
    tf.setUseSet("test")
    prices = timeSeries.get_prices(use_set = "trainTest")
    prices_test = timeSeries.get_prices(use_set = "test")
    returns = timeSeries.get_returns(use_set = "trainTest")
    returns_test = timeSeries.get_returns(use_set = "test")
    
    buy_hold_test = cash_start / prices_test[0] * prices_test
    tf_sma_test = tf.TF_simple(best_tf_sma_period)
    tf_ema_test = tf.TF_exponential(best_tf_ema_alpha)
    tf_cross_sma_test = tf.TF_crossover_simple(best_tf_cross_sma_long_period, best_tf_cross_sma_short_period)
    tf_cross_ema_test = tf.TF_crossover_exponential(best_tf_cross_ema_long_alpha, best_tf_cross_ema_short_alpha)
    tf_bb_rsi_test = tf.TF_bb_rsi(best_tf_bb_period, best_tf_bb_std, best_tf_rsi_period)
    
    figure, axis = plt.subplots(nrows = 2, ncols = 1, figsize = (7, 7))
    
    TimeSeries.plot([("Buy and Hold", buy_hold_test), 
                                    ("TF SMA [" + str(best_tf_sma_period) + "]" , tf_sma_test),
                                    ("TF EMA [" + str(best_tf_ema_alpha) + "]" , tf_ema_test),
                                    ("TF CROSS SMA [" + str(best_tf_cross_sma_long_period) + ", " + str(best_tf_cross_sma_short_period) + "]" , tf_cross_sma_test),
                                    ("TF CROSS EMA [" + str(best_tf_cross_ema_long_alpha) + ", " + str(best_tf_cross_ema_short_alpha) + "]" , tf_cross_ema_test),
                                    ("TF BB RSI [" + str(best_tf_bb_period) + "]", tf_bb_rsi_test)],
                                    "Strategies Test Set", "Days", "Cash", axis = axis[0])
    axis[0].xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    print("Buy Hold test is", buy_hold_test[-1])
    print("TF SMA test is", tf_sma_test[-1])
    
    tf.timeSeries.reports.writeReportTest("")
    plt.show()
