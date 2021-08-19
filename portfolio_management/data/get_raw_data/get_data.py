import pandas as pd 
from yahoofinancials import YahooFinancials

class GetYFArchive():  
    @classmethod
    def data_dict(cls,raw_data,symbols):
        data = {}

        for symbol in symbols:
            try:
                prices = pd.DataFrame(raw_data[symbol]['prices'])
            except:
                continue
            # date field is raw, we drop it.
            try:
                 prices.drop('date', axis = 1,inplace=True)
            except:
                continue
            # Formatted date is string "yyyy-mm-dd", we convert to datetime format
            try:
                prices['Date'] = pd.to_datetime(prices['formatted_date'])
            except:
                continue
            # Drop Date string and set Date to index.
            prices.drop(columns = ['formatted_date'], inplace = True)
            prices.set_index(['Date'],inplace = True)
 
            # Add prices DataFrame to dictionary
            data[symbol] = prices
        
            data[symbol].fillna(0,inplace = True)
            data[symbol].insert(0,'Symbol',symbol)

        return data
        
    @classmethod    
    def get(cls,start,end,symbols,freq): 
        #symbols = symbols[:10]
        yf = YahooFinancials(symbols)
        raw_data = yf.get_historical_price_data(start_date=start, 
                                                  end_date=end, 
                                                  time_interval=freq)
        
        data = cls.data_dict(raw_data,symbols)
        
        stock_data = pd.concat([data[symbol] for symbol in data])
        stock_data.reset_index(inplace = True)
        
        return stock_data
  