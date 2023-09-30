from imports import *
from execute.algoExecute import *
from timeSeries.TimeSeries import TimeSeries
from timeSeries.BinanceTimeSeries import BinanceTimeSeries


def getExecuteStrategyReturns(client, ordersPath):
    orders = pd.read_csv(ordersPath)
    initialOrder = orders.iloc[0]
    firstOrder = orders.iloc[1] # first row is the hold and not an actual order
    symbol = "BTCUSDT"
    interval = Client.KLINE_INTERVAL_1HOUR
    untilThisDate = datetime.datetime.now()
    sinceThisDate = datetime.datetime.fromtimestamp(firstOrder["time"] // 1000, tz = datetime.timezone.utc) - datetime.timedelta(hours = 2)
    timeSeries = BinanceTimeSeries(client, symbol, sinceThisDate, untilThisDate, interval, 0)
        
    origPrices = timeSeries.get_prices(use_set = "train").copy()
    prices = origPrices.copy()
    w = pd.Series(data = np.nan, index = prices.index)
    cash = pd.Series(data = np.nan, index = prices.index)
    
    # works because we subtract 2 hours
    w.iloc[0] = initialOrder["btc"]
    cash.iloc[0] = initialOrder["usdt"]
    for i in range(1, len(orders)):
        time = pd.Timestamp(orders.iloc[i]["time"], unit = "ms").replace(minute = 0).replace(second = 0).replace(microsecond = 0)
        prices[time] = orders.iloc[i]["price"]
        w[time] = orders.iloc[i]["btc"]
        cash[time] = orders.iloc[i]["usdt"]
    
    for i in range(1, len(prices)):
        if not np.isnan(w.iloc[i]):
            continue
        
        w.iloc[i] = w.iloc[i - 1]
        cash.iloc[i] = cash.iloc[i - 1]
        
    # BUY AND HOLD
    cashStart = initialOrder["usdt"]
    buyHold = cashStart / prices[0] * prices
    
    strategyReturns =  [a * b for a, b in zip(w, prices)] + cash
    
    return origPrices, buyHold, strategyReturns
    

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("config/prod.env")
    client = Client(config["BINANCE"]["bin_api_key"], config["BINANCE"]["bin_api_secret"])
    tfPrices, tfBuyHold, tfOrdersReturns = getExecuteStrategyReturns(client, "tfOrders.csv")
    mrPrices, mrBuyHold, mrOrdersReturns = getExecuteStrategyReturns(client, "mrOrders.csv")
    figure, axis = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 7))
    
    axis[0,0].plot(tfPrices)
    TimeSeries.plot([("Buy and Hold", tfBuyHold), 
                            ("MR Execute", tfOrdersReturns)],
                            "TF Test Set", "Days", "Cash", axis = axis[1,0])
    axis[1,0].xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    axis[0,1].plot(mrPrices)
    TimeSeries.plot([("Buy and Hold", mrBuyHold), 
                            ("MR Execute", mrOrdersReturns)],
                            "MR Test Set", "Days", "Cash", axis = axis[1,1])
    axis[1,1].xaxis.set_major_locator(ticker.MaxNLocator(5))
    
    plt.show()
