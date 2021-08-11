import pandas as pd 
from yahoofinancials import YahooFinancials

class GetYFArchive():  
    @classmethod
    def data_dict(cls,raw_data,symbols):
        data = {}

        for symbol in symbols:
            
            prices = pd.DataFrame(raw_data[symbol]['prices']).drop('date', axis = 1)
            prices['Date'] = pd.to_datetime(prices['formatted_date'])
            prices.drop(columns = ['formatted_date'], inplace = True)
            prices.set_index(['Date'],inplace = True)
    

            data[symbol] = prices
        
            data[symbol].fillna(0,inplace = True)
            data[symbol].insert(0,'Symbol',symbol)

        return data
        
    @classmethod    
    def get(cls,start,end,symbols,freq): 
        
        yf = YahooFinancials(symbols)
        raw_data = yf.get_historical_price_data(start_date=start, 
                                                  end_date=end, 
                                                  time_interval=freq)
        
        data = cls.data_dict(raw_data,symbols)
        
        stock_data = pd.concat([data[symbol] for symbol in data])
        stock_data.reset_index(inplace = True)
        
        return stock_data
  