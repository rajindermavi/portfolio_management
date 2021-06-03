
import gym
from gym import spaces
#from gym.utils import seeding
#import pandas as pd
import numpy as np

class TradingEnv(gym.Env):

    '''
    Create trading environment our agent will interact with.

    stocks: numpy array for stock data over time
    state: numpy array for econ data over time
    window_size: number of past timeslots our agent has access to when making a trade decision.
    trading_cost: cost for executing a trade
    interest_rate: risk-free intrest rate
    portfolio_value: initial value of portfolio
    training_size: fraction of data used for training
    '''
    metadata = {'render.modes': ['human']}
    def __init__(self,stock_data,
                 window_size=64,trading_cost=1e-4,
                 interest_rate=1e-6,portfolio_value=1e6,
                 training_size=.8):
        
        #data
        self.stock_data = stock_data
        #parameters
        self.window_size=window_size
        self.trading_cost=trading_cost
        self.interest_rate=interest_rate
        self._initial_portfolio_value=portfolio_value

        #dimensions of data
        #self.n_econ_feats = self.econ_data.shape[1]

        self.n_stocks = self.stock_data.shape[0]
        self.n_stock_feats = self.stock_data.shape[2]
        self.total_data = self.stock_data.shape[1]

        #episode
        self._start_tick = self.window_size-1
        self._end_tick = int(self.total_data*training_size)
        self._episode_length = self._end_tick - self.window_size

        self._done = False
        self._idx = None

        #portfolio
        initial_array = [self._initial_portfolio_value]+self.n_stocks*[0]
        self._initial_portfolio = np.array(initial_array)
 
        self._portfolio_weights = None
        self._portfolio_value = None
        self.portfolio_value_hist = None
        self.rewards_hist = None
        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_stocks + 1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape = (self.n_stocks,self.window_size,self.n_stock_feats),
            dtype=np.float32
        )


    def reset(self):
        '''
        Restarts environment for training.

        Returns initial stock and data information. 
        '''
        self._done = False
        self._idx = self._start_tick
        self._portfolio = self._initial_portfolio
        self._portfolio_value = self._initial_portfolio_value
        self._portfolio_weights = self._portfolio_weight_eval()
        self.portfolio_value_hist = self.window_size * [self._initial_portfolio_value]
        self.rewards_hist = self.window_size * [0]

        return self._get_observation()

    def step(self,action):
        '''
        Updates portfolio with action. 
        action: weights of new portfolio.

        Reward is defined as the log return of the porfolio.
        '''
        # prestep values 
        prices_prestep = self.stock_data[:,self._idx,0]
        prestep_portfolio_value = self._portfolio_value
        w_prestep = self._portfolio_weights

        # refinancing
        w_refi = action.flatten()
        #  - refi is approximated to simplify calculations
        #  - note w[0] is cash so moving value in and out of that account does not charge
        refi_volume = prestep_portfolio_value*np.linalg.norm(w_refi[1:]-w_prestep[1:])
        refi_portfolio_value = prestep_portfolio_value - self.trading_cost*refi_volume

        # time step 
        self._idx += 1
        
        if self._idx >= self._end_tick:
            self._done = True

        #poststep values 
        prices_poststep = self.stock_data[:,self._idx,0]

        # value / weight evolution
        cash_ratio = np.array([1])
        y_evol = np.concatenate((cash_ratio, prices_poststep/prices_prestep))
        w_evol = (y_evol*w_refi)/(np.dot(y_evol,w_refi))

        # record the new portfolio value
        poststep_portfolio_value = refi_portfolio_value*np.dot(y_evol,w_refi)
        self._portfolio_value = poststep_portfolio_value
        self.portfolio_value_hist.append(self._portfolio_value)
        self._portfolio_weights = w_evol

        # calculate reward 
        reward = np.log(poststep_portfolio_value / prestep_portfolio_value)
        info = []

        return self._get_observation(), reward, self._done, info

    def _portfolio_weight_eval(self):
        '''
        Returns the weight vector of the current portfolio.
        '''
        return self._portfolio/self._portfolio_value  

    def _get_observation(self):    
        '''
        Return window of normalized stock data and current econ data.
        '''

        window_stock_data = self.stock_data[:, self._idx-self.window_size+1:self._idx+1]
        #Normalize by timed prices:
        X = window_stock_data
        X = X.transpose(2,1,0)
        X = X / X[0,0,:]
        X = X.transpose(2,1,0)

        #econ_data = self.econ_data[self._idx]
        #econ_data = self.econ_data[0]
        return X#, econ_data





