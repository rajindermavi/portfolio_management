from agents import agent_loss, CAPM_Agent, MVP_Agent, Uniform_Agent
from trading_env.environment import TradingEnv
from data.get_raw_data import GetYFArchive
import numpy as np
import dill 

class MetaAgent():
    def __init__(self,n_stocks):
        self.n_stocks = n_stocks
        self.create_agents()

    def create_agents(self): 
        with open('dpm_agent.dill','rb') as dill_file:
            self.dpm_agent = dill.load(dill_file)
        self.capm_agent = CAPM_Agent(self.n_stocks)
        self.mvp_agent = MVP_Agent(self.n_stocks)
        self.uniform_agent = Uniform_Agent(self.n_stocks)

class MetaEnv():
    def __init__(self,symbols,start_date,end_date):
        self.symbols=symbols
        self.start_date=start_date
        self.end_date=end_date
        # Fetch data online as pandas DataFrame
        self.yf_data=self.get_yf_data() 
        # Convert dataframe to numpy array
        self.dates, self.yf_array=self.yf_data_to_array(self.yf_data)
        
        self.env=TradingEnv(self.yf_array)

    def get_yf_data(self):
        yf=GetYFArchive()
        raw_data=yf.get(self.start_date,self.end_date,self.symbols,'daily')
        return [raw_data[raw_data['Symbol']==symbol]for symbol in self.symbols]

    def yf_data_to_array(self,data):
        dates=data[0]['Date']
        array=np.array([df.iloc[:,2:6].to_numpy() for df in data])
        return dates, array 

class AgentComparison():
    def __init__(self,symbols,start_date='2021-01-01',end_date='2021-06-01'):

        self.meta_env=MetaEnv(symbols,start_date,end_date)
        self.n_stocks = self.meta_env.env.n_stocks 
        self.meta_agent = MetaAgent(self.n_stocks)

        self.dpm_vals,self.uniform_vals,self.mvp_vals,self.capm_vals=self.simulate_agents()

    def simulate_agents(self):

        start_idx = self.meta_env.env._start_tick

        _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.dpm_agent,
                           self.n_stocks)
        dpm_val_hist = self.meta_env.env.portfolio_value_hist

        _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.uniform_agent,
                           self.n_stocks)
        uniform_val_hist = self.meta_env.env.portfolio_value_hist

        _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.mvp_agent,
                           self.n_stocks)
        mvp_val_hist = self.meta_env.env.portfolio_value_hist

        _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.capm_agent,
                           self.n_stocks)
        capm_val_hist = self.meta_env.env.portfolio_value_hist

        return dpm_val_hist,uniform_val_hist,mvp_val_hist,capm_val_hist