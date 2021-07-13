from agents import agent_loss, CAPM_Agent, MVP_Agent, Uniform_Agent
from trading_env.environment import TradingEnv
from data import GetYFArchive
import plotly.graph_objects as go
import numpy as np
import dill 
import os

class MetaAgent():
    def __init__(self,n_stocks):
        self.n_stocks = n_stocks
        self.dpm_agent,self.capm_agent,self.mvp_agent,self.mvp_agent=None,None,None,None
        self.create_agents()
        self.agent_dic={'DeepPortfolio':self.dpm_agent,
                        'CAPM':self.capm_agent,
                        'MVP':self.mvp_agent,
                        'Uniform':self.uniform_agent}

    def create_agents(self): 
        dirname=os.path.dirname(__file__)
        filename=os.path.join(dirname,'dpm_agent.dill')
        with open(filename,'rb') as dill_file:
            self.dpm_agent = dill.load(dill_file)
        self.capm_agent = CAPM_Agent(self.n_stocks)
        self.mvp_agent = MVP_Agent(self.n_stocks)
        self.uniform_agent = Uniform_Agent(self.n_stocks)

class MetaEnv():
    def __init__(self,yf_archive,date_mask,select_symbols):

        self.yf_archive = yf_archive
        #self.range_start = range_start
        #self.range_end = range_end
        self.date_mask = date_mask
        self.select_symbols = select_symbols
        
        # subset the archive data according to given parameters
        self.yf_data = self.subset_data()

        # Convert dataframe to numpy array
        self.trading_dates, self.yf_array=self.yf_data_to_array(self.yf_data)
        
        self.env=TradingEnv(self.yf_array)

    def subset_data(self):
        '''Return a list of pandas DFs for each symbol in the selected date ranges.'''
        # list of masks to subset symbols
        symbol_masks = [self.yf_archive['Symbol']==symbol for symbol in self.select_symbols]
        # Subset and return data according to masks
        return [self.yf_archive[self.date_mask &symbol_mask] for symbol_mask in symbol_masks]

    def yf_data_to_array(self,data):
        trading_dates=data[0]['Date']
        array=np.array([df.iloc[:,2:6].to_numpy() for df in data])
        return trading_dates, array 

class AgentComparison():
    def __init__(self,agents,symbols,start_date='2019-06-01',end_date='2021-06-01'):

        self.agents = agents
        self.meta_env = MetaEnv(symbols,start_date,end_date)
        self.n_stocks = self.meta_env.env.n_stocks 
        self.meta_agent = MetaAgent(self.n_stocks)
        self.agents_vals = []

    def simulate_agent(self,agent,x,symbols):

        _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.agent_dic[agent],
                           self.n_stocks)
        self.agents_vals.append(self.meta_env.env.portfolio_value_hist)


    def simulate_agent(self):

        if 'DeepPortfolio' in self.agents:
            _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.dpm_agent,
                           self.n_stocks)
        self.agents_vals.append(self.meta_env.env.portfolio_value_hist)    
        
        if 'CAPM' in self.agents:
            _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.uniform_agent,
                           self.n_stocks)
            self.agents_vals.append(self.meta_env.env.portfolio_value_hist)

        if 'MVP' in self.agents:
            _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.mvp_agent,
                           self.n_stocks)
            self.agents_vals.append(self.meta_env.env.portfolio_value_hist)

        if 'Uniform' in self.agents:
            _loss = agent_loss(self.meta_env.env,
                           self.meta_agent.capm_agent,
                           self.n_stocks)
            self.agents_vals.append(self.meta_env.env.portfolio_value_hist)

    def plot_agent_histories(self):
        start = self.meta_env.env._start_tick
        date_range=self.meta_env.dates[start:]
        fig = go.Figure()
        for idx,agent in enumerate(self.agents):
            fig.add_trace(go.Scatter(x=date_range,
                          y=self.agents_vals[idx],
                          mode='lines',name=agent)) 
        return fig,date_range
