import os
import pandas as pd
import dill
import json
import os
import datetime as dt
from pathlib import Path

import get_raw_data
import data_processing

class GetArchive():
    def __init__(self,datasets=['fred','yahoo','nyt'],to_save=True):
        
        self.fred_config,self.nyt_config,self.yf_config=None,None,None
        print('Loading configuration files.')
        self.load_configs()
        print('Creating archives.')
        self.archives()
        
        self.fred_data=None
        self.nyt_data=None
        self.yf_data=None


        self.get_datasets(datasets) 
             
             
    def load_configs(self):
        dirname=os.path.dirname(__file__)
        configs_file = os.path.join(dirname,'get_raw_data','configs','get_raw_data_configs.json')
        with open(configs_file,'r') as cf:
            configs = json.load(cf)
            self.fred_config = configs['fred']
            self.yf_config = configs['yahoo']
            self.nyt_config = configs['nyt']
            
        api_keys_file = os.path.join(dirname,'get_raw_data','configs','api_keys.json')    
        with open(api_keys_file,'r') as apis:
            keys = json.load(apis)
            self.fred_api = keys['fred']
            self.nyt_api = keys['nyt'] 
        
    def archives(self):
        '''Creates instances of archive readers'''
        self.fred_archive = get_raw_data.GetFredArchive(self.fred_api)
        self.yf_archive = get_raw_data.GetYFArchive()
        self.nyt_archive = get_raw_data.GetNYTArchive(self.nyt_api)
        
    def get_datasets(self,datasets):
        '''Uses config data to retrieve data from archives.'''
        if 'fred' in datasets:
            print("Getting FRED data.")
            self.get_fred()
        if 'yahoo' in datasets:
            print("Getting Yahoo Financial data.")
            self.get_yf()
        if 'nyt' in datasets:
            print("Getting New York Times data.")
            self.get_nyt() 
        
    def get_fred(self,start=None,end=None,series_collection=None):
        start = self.fred_config['start']
        end = self.fred_config['end']
        series_collection = self.fred_config['series_collection']
        
        self.fred_data = self.fred_archive.get(start,end,series_collection)

    def get_yf(self,start=None,end=None,symbols=None,freq=None):
        start = self.yf_config['start_date']
        end = self.yf_config['end_date']
        symbols = self.yf_config['symbols']
        freq = self.yf_config['freq']
        
        self.yf_data = self.yf_archive.get(start,end,symbols,freq)
        
    def get_nyt(self,year_start=None,year_end=None):
        year_start,year_end = self.nyt_config['year_start'],self.nyt_config['year_end']
        
        self.nyt_data = self.nyt_archive.get(year_start,year_end)
                
class DataPrep:
    @staticmethod        
    def feat_engineering(stock_series):
        '''Takes pd series of stock data and returns constructed features'''
        
        features = {}
        features['vbm'] = data_processing.vbm(stock_series)
        features['std'] = data_processing.std(stock_series)
        features['hurst'] = data_processing.hurst(stock_series)
        features['stoch_osc'] = data_processing.stoch_osc(stock_series)
        features['force'] = data_processing.force(stock_series)
        features['volume_ratio'] = data_processing.volume_ratio(stock_series)
        features['price_ratio'] = data_processing.price_ratio(stock_series)
        
        feat_df = pd.concat([val for key, val in features.items()],axis = 1)
        feat_df.columns = features.keys()  
        feat_df.dropna(inplace = True)
        
        return feat_df
    @staticmethod    
    def vectorize_news(news_df):
        '''Takes in news data frame and converts it to sequence of numerical vectors'''
        
        nle = data_processing.NLEncoding()
        vec = nle.vectorize(news_df.News)
        
        return pd.DataFrame(data = vec,index = news_df.index) 
    @staticmethod    
    def join(news_vecs_df,feat_df,fred_df):
        '''Takes constructed features and econ features and creates a single dataframe'''
        #self.stock_df = pd.read_pickle('.data_stocks.pkl')
        #self.econ_df = pd.read_pickle('.data_econ.pkl')
        
        dfs = [fred_df,news_vecs_df]
        
        return feat_df.join([dfs], on = 'Date', how = 'inner')

if __name__ == "__main__":
      
    archive = GetArchive(datasets=['yahoo'])
    
    save_dir = './archive_data'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    #time = dt.datetime.now().strftime('%d%m%Y_%H%M')
    if type(archive.nyt_data) == pd.DataFrame:
        nyt_file = 'nyt_data.dill'
        save_nyt = os.path.join(save_dir, nyt_file)
        with open(save_nyt, "wb") as dill_file:
            dill.dump(archive.nyt_data, dill_file)
    if type(archive.fred_data)==pd.DataFrame:    
        fred_file = 'fred_data.dill'
        save_fred = os.path.join(save_dir, fred_file)
        with open(save_fred, "wb") as dill_file:
            dill.dump(archive.fred_data, dill_file)
    if type(archive.yf_data)==pd.DataFrame:
        yf_file = 'yf_data.dill'
        save_yf = os.path.join(save_dir, yf_file)
        with open(save_yf, "wb") as dill_file:
            dill.dump(archive.yf_data, dill_file)
    
    
    