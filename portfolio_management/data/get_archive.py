import os
import dill
import json 
from pathlib import Path
import get_raw_data 

class GetArchive():
    def __init__(self):
        
        self.yf_config=None 
        print('Loading configuration files.')
        self.load_configs()
        print('Creating archive.')
        self.yf_archive = get_raw_data.GetYFArchive()
        
        print("Getting Yahoo Financial data.")
        self.yf_data=None
        self.get_yf()
             
             
    def load_configs(self):
        configs_file = 'yahoo.json'
        with open(configs_file,'r') as cf:
            configs = json.load(cf) 
            self.yf_config = configs

    def get_yf(self):
        start = self.yf_config['start_date']
        end = self.yf_config['end_date']
        symbols = self.yf_config['symbols']
        freq = self.yf_config['freq']
        
        self.yf_data = self.yf_archive.get(start,end,symbols,freq)

if __name__ == "__main__":
      
    archive = GetArchive()
    
    # Set save location
    save_dir = './archive_data'
    Path(save_dir).mkdir(parents=True, exist_ok=True) 
    yf_file = 'yf_data.dill'
    save_yf = os.path.join(save_dir, yf_file)
    print(f'Saving yahoo finance archive at {save_yf}.')
    with open(save_yf, "wb") as dill_file:
        dill.dump(archive.yf_data, dill_file)

    symbols = archive.yf_data['Symbol'].unique()

    print(symbols)
    print(f'Number of symbols: {len(symbols)}')
    
    