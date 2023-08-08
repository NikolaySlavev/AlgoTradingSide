from imports import *
from strategies.ARIMA import ForecastArimaGarch
from timeSeries.TimeSeries import TimeSeries
from timeSeries.BinanceTimeSeries import BinanceTimeSeries
from statistics_1 import *


if __name__ == "__main__":
    client = Client(bin_api_key, bin_api_secret)
    cash_start = 100
    numSplits = 2
    symbol = "BTCUSDT"
    days = 100
    interval = Client.KLINE_INTERVAL_1HOUR
    
    start_time = time.time()
    timeSeries = BinanceTimeSeries.fromHowLong(client = client, dataPair = symbol, howLong = days, interval = interval, numSplits = numSplits)
    timeSeries.set_current_train_test_data(1)
    differencingPlots(timeSeries.get_prices(use_set = "train"))
    
    prices = timeSeries.get_prices(use_set = "trainTest")
    prices_test = timeSeries.get_prices(use_set = "test")
    returns = timeSeries.get_returns(use_set = "trainTest")
    returns_train = timeSeries.get_returns(use_set = "train")
    returns_test = timeSeries.get_returns(use_set = "test")

    # BUY AND HOLD
    buy_hold = cash_start / prices[0] * prices
    buy_hold_train = buy_hold[:len(returns_train)]
    buy_hold_test = cash_start / prices_test[0] * prices_test

    # ARIMA
    arima_garch_model = ForecastArimaGarch(timeSeries, cash_start)
    arima_train, arima_pred_train, arima_order_train = arima_garch_model.ARIMA_GARCH_train()
    print(arima_order_train)
    arima_test, arima_pred_test, arima_buy_sell_test, arima_order_test = arima_garch_model.ARIMA_GARCH_test()

    figure, axis = plt.subplots(nrows = 2, ncols = 2, figsize=(10, 7))
    axis[0, 0].plot(prices)
    axis[0, 0].plot(prices_test, color = "orange")
    axis[0, 0].xaxis.set_major_locator(ticker.MaxNLocator(5))
    axis[0, 0].set_title("Prices")
    axis[0, 1].plot(returns)
    axis[0, 1].plot(returns_test, color = "orange")
    axis[0, 1].xaxis.set_major_locator(ticker.MaxNLocator(5))
    axis[0, 1].set_title("Returns")
    
    TimeSeries.plot([("Buy and Hold", buy_hold_train), 
                            ("ARIMA+GARCH" + str(arima_order_train), arima_train)], 
                            "Strategies Train Set", "Days", "Cash", axis = axis[1, 0])
    axis[1, 0].xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    TimeSeries.plot([("Buy and Hold", buy_hold_test), 
                            ("ARIMA+GARCH" + str(arima_order_test), arima_test)], 
                            "Strategies Test Set", "Days", "Cash", axis = axis[1, 1])
    axis[1, 1].xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    plt.show()
