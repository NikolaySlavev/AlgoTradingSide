from imports import *
from TimeSeries import TimeSeries


"""
    Class to create random Synthetic Time Series.
    The parameters are preset to meet the requirements, but can be adjusted if needed
    Seed can be specified to get the same Time Series
"""
class SyntheticTimeSeries(TimeSeries):
    # Create the Time Series
    def __init__(self, seed = 1, t = 3000, phi = 0.5, d = 0.02, theta = -0.3, mean = 0, variance = 1, p0 = 1000, p1 = 1000, train_test_split = 0.7):
        self.train_test_split = train_test_split
        self.df = SyntheticTimeSeries.generate_data(seed = seed, t = t, phi = phi, d = d, theta = theta, mean = mean, variance = variance, p0 = p0, p1 = p1)        
    
    def generate_data(seed, t, phi, d, theta, mean, variance, p0, p1):
        np.random.seed(seed)
        series = [p0, p1]
        change = p1 - p0
        eps =   np.random.normal(mean, variance, t)
        for i in range(1, t-1):
            change_prev = change
            change = phi * (change_prev - d) + eps[i] + theta * eps[i-1] + d                
            series.append(series[-1] + change)
        
        df = pd.DataFrame()
        df[PRICE] = pd.Series(data = series)
        return df
        
