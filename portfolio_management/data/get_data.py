import pandas as pd 
from yahoofinancials import YahooFinancials
      
class GetYFArchive(): 
    def __init__(self):  
        #self.configs = json.load(open('.configs/yahoo_financial.json', 'r')) 
        pass
    
    def data_dict(self,raw_data,symbols):
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
        
    def get(self,start,end,symbols,freq):
        #start = self.configs_stock['start_date']
        #end = self.configs_stock['end_date']
        #symbols = self.configs_stock['symbols']
        #freq = self.configs_stock['freq']
        
        yf = YahooFinancials(symbols)
        raw_data = yf.get_historical_price_data(start_date=start, 
                                                  end_date=end, 
                                                  time_interval=freq)
        
        data = self.data_dict(raw_data,symbols)
        
        stock_data = pd.concat([data[symbol] for symbol in data])
        #stock_data['Date'] = stock_data.index
        stock_data.reset_index(inplace = True)
        
        return stock_data
   