import pandas as pd
import numpy as np 

def attribute(df, attr):
    '''Create dataframe with attr columns and Date rows.'''
    assert isinstance(df, pd.DataFrame), print('First argument must be a pandas dataframe.')
    assert set([attr]).issubset(set(df.columns)), print('Dataframe is missing necessary columns')
    Symbols = df.Symbol.unique() 
    
    result_list = [df[df.Symbol == s].loc[:,['Date',attr]].set_index('Date') for s in Symbols]
    result_df = pd.concat(result_list,axis = 1)
    result_df.columns = Symbols
    
    return result_df
    
def vbm(df, v = 65, d = 22,eps = 0.01):
    '''Volatility based momentum.'''
    assert isinstance(df, pd.DataFrame), print('First argument must be a pandas dataframe.')
    assert set(['Symbol','Date']).issubset(set(df.columns)), print('Dataframe is missing necessary columns')
 
    highs = attribute(df, 'high')
    lows = attribute(df, 'low')
    prices = attribute(df, 'close')
   
    log_range = highs.apply(np.log) - lows.apply(np.log)
    
    
    price_diff = prices.apply(lambda x: np.log(x) - np.log(x.shift(d)))
    log_range_mean = eps + d*log_range.rolling(v).mean()
    
    return log_range_mean
    #return price_diff /log_range_mean

def std(df, v = 65):
    '''Standard deviation of price changes over last v days for each symbol in df'''
    assert isinstance(df, pd.DataFrame), print('First argument must be a pandas dataframe.')
    assert set(['Symbol','Date']).issubset(set(df.columns)), print('Dataframe is missing necessary columns')
    
    prices = attribute(df, 'close')
    
    return prices.apply(lambda x: np.log(x) - np.log(x.shift(1))).rolling(v).std()   
    
def hurst(df,v = 200, d = 20): 
    '''Calculate Hurst exponent.'''
    assert d >= 10, print('Range of lags must be d>= 10')
    assert v >= 5*d, print('Window must be at least 5 times range of lags')
    assert isinstance(df, pd.DataFrame), print('First argument must be a pandas dataframe.')
    assert set(['Symbol','Date']).issubset(set(df.columns)), print('Dataframe is missing necessary columns')
    
    prices = attribute(df, 'close')
    
    def window_hurst(w):
        lags = range(2, d)
        tau = [np.sqrt(np.std(np.subtract(w[lag:], w[:-lag]))) for lag in lags]
 
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        return m[0]*2.0
    
    return prices.rolling(v).apply(window_hurst, raw = True)

def stoch_osc(df,p = 5):
    highs = attribute(df, 'high')
    lows = attribute(df, 'low')
    prices = attribute(df, 'close')
    
    lows_p = lows.rolling(p).min()
    highs_p = highs.rolling(p).max()
    
    stoch_osc_p = (prices.apply(np.log) - lows_p.apply(np.log))/(highs_p.apply(np.log) - lows_p.apply(np.log))

    return stoch_osc_p.apply(lambda x: (x+x.shift(p) + x.shift(2*p))/3 )

def force(df, p = 15):

    prices = attribute(df, 'close')
    volumes = attribute(df, 'volume')
    
    force_raw = (prices.apply(lambda x: np.log(x) - np.log(x.shift(1))) 
                 * volumes.apply(np.log))
    
    return force_raw.ewm(halflife = p).mean()

def volume_ratio(df, v = 65, d = 20):
    assert isinstance(df, pd.DataFrame), print('First argument must be a pandas dataframe.')
    assert set(['Symbol','Date']).issubset(set(df.columns)), print('Dataframe is missing necessary columns')
    
    volumes = attribute(df, 'volume')
    
    return volumes.apply(np.log).rolling(d).mean()/volumes.apply(np.log).rolling(v).mean()

def price_ratio(df, v = 65, d = 20):
    assert isinstance(df, pd.DataFrame), print('First argument must be a pandas dataframe.')
    assert set(['Symbol','Date']).issubset(set(df.columns)), print('Dataframe is missing necessary columns')
    
    prices = attribute(df, 'close')
    
    return prices.apply(np.log).rolling(d).mean()/prices.apply(np.log).rolling(v).mean()

