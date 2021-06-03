import pandas as pd 
from fredapi import Fred 
from yahoofinancials import YahooFinancials
import requests
import datetime
import time

class GetFredArchive(): 
    def __init__(self,API_KEY):  
        #self.configs = json.load(open('./configs/fred.json', 'r')) 
        #self.api = json.load(open('./configs/fred_api.json', 'r'))[0]
        self.API_KEY = API_KEY
        
    def get(self,start,end,series_collection):
        #start = self.configs['start']
        #end = self.configs['end']
        #series_collection = self.configs['series_collection']
        
        fred = Fred(api_key=self.API_KEY)
        
        econ_data = {} 

        for series in series_collection:
            fred_series = fred.get_series(series,observation_start=start,observation_end=end) 
            econ_data[series] = fred_series
        
        fred_df = pd.concat([econ_data[series] for series in series_collection],axis=1,join='outer')
        fred_df.columns = series_collection
        fred_df = fred_df.fillna(method='ffill').fillna(method='bfill')
        fred_df.rename_axis(index = 'Date',inplace = True)
        
        return fred_df
      
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
   
class GetNYTArchive(): 
    def __init__(self,API_KEY):  
        #self.configs = json.load(open('.configs/nyt_news.json', 'r'))  
        #self.api = json.load(open('./configs/nyt_api.json', 'r'))[0]
        self.API_KEY = API_KEY      
            
    def get_url(self,year,month):
        base_url = "http://api.nytimes.com/svc/archive/v1/"
        query_url = f'{year}/{month}.json?api-key={self.API_KEY}'
        return base_url + query_url
        
    def archive_query(self,year,month):
        '''Returns list of articles per month with meta data'''
     
        url = self.get_url(year,month)
        r = requests.get(url).json()

        return r['response']['docs']            
            
    def query_df(self,year,month):
        '''Returns dataframe of business articles on first 3 pages per month, date and abstract'''
        rows = []
    
        articles = self.archive_query(year,month)
    
        for article in articles: 
            try:
                #desk = article['news_desk']
                #ps = article['print_section']
                #pp = article['print_page']
                #if desk in ['Business'] and ps == 'A' and pp in '123':  
                rows.append({'Date':article['pub_date'] ,
                             'Desk':article['news_desk'],
                             'Page':article['print_page'],
                             'Section':article['print_section'],
                             'Abstract': article['abstract'],
                             'Lead':article['lead_paragraph'],
                                }) 
            except:
                pass       
    
        timeconv = lambda x: datetime.datetime.strptime(x[:10], '%Y-%m-%d')
        data = pd.DataFrame(rows) 
        data.Date = data.Date.apply(timeconv)    
        return data
    
    def archive_df(self,years,months):
    
        df = pd.DataFrame()
    
        for year in years:
            for month in months:
                time.sleep(7)
                df = pd.concat([df,self.query_df(year,month)])
        df.reset_index(drop = True, inplace = True)
        return df
             
    def get(self,year_start,year_end):
        assert year_start <= year_end and year_end < 2021, print('Improper date range.')
        years = list(range(year_start,year_end+1))
        months = list(range(1,13)) 

        return (self.archive_df(years,months)).set_index('Date')