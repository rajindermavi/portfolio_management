
import gym
from gym import spaces
import numpy as np
import tensorflow as tf

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
                 interest_rate=1e-5,portfolio_value=1e6):
        
        #data
        self.stock_data = tf.convert_to_tensor(stock_data,
                                            dtype=np.float32)
        # Data has dimensions:
        # 0: Stocks
        # 1: Time series
        # 2: Features: (high, low, open, close)

        # Use closing prices for trading values:
        self.price_idx = -1

        #parameters
        self.window_size=window_size
        self.trading_cost=trading_cost
        self.interest_rate=interest_rate
        self._initial_portfolio_value=tf.convert_to_tensor(portfolio_value)

        #dimensions of data
        #self.n_econ_feats = self.econ_data.shape[1]

        self.n_stocks = self.stock_data.shape[0]
        self.n_stock_feats = self.stock_data.shape[2]
        self.total_data = self.stock_data.shape[1]

        # Create a tensor for time value of money
        #  and adjoin it to stock data array.
        money = self.money()
        self.stock_data = tf.concat([self.stock_data,money],axis=0)

        #episode
        self._start_tick = self.window_size-1
        self._end_tick = int(self.total_data) -1
        self._episode_length = self._end_tick - self.window_size

        self._done = False
        self._idx = None

        #portfolio
        initial_array = [self._initial_portfolio_value]+self.n_stocks*[0]
        self._initial_portfolio = tf.convert_to_tensor([initial_array])
 
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
    
    def money(self):
        '''Constructs time value of money array'''
        money = np.exp(self.interest_rate * 
                       np.linspace(0,self.total_data-1,self.total_data))
        money = np.transpose(np.array([self.n_stock_feats * [money]]),
                            (0,2,1))

        return tf.convert_to_tensor(money,dtype=np.float32)

    def reset(self):
        '''
        Restarts environment for training.

        Returns initial stock and data information. 
        '''
        self._done = False
        self._idx = self._start_tick
        self._portfolio = self._initial_portfolio
        self._portfolio_value = self._initial_portfolio_value
        self._portfolio_weights = self._portfolio/self._portfolio_value  
        self.portfolio_value_hist = [self._initial_portfolio_value]
        self.rewards_hist = [0]

        return self._get_observation()

    def step(self,action):
        '''
        Updates portfolio with action. 
        action: weights of new portfolio.

        Reward is defined as the log return of the porfolio.
        '''
        # prestep values 
        prices_prestep = tf.convert_to_tensor([self.stock_data[:,self._idx,self.price_idx]])
        prestep_portfolio_value = self._portfolio_value
        w_prestep = self._portfolio_weights

        # refinancing
        w_refi = action
        #  - refi is approximated to simplify calculations
        #  - note w[0] is cash so moving value in and out of that account does not charge
        refi_volume = prestep_portfolio_value*tf.norm(w_refi[1:]-w_prestep[1:],ord=1)
        refi_portfolio_value = prestep_portfolio_value - self.trading_cost*refi_volume

        # time step 
        self._idx += 1
        
        if self._idx >= self._end_tick:
            self._done = True

        #poststep values 
        prices_poststep = tf.convert_to_tensor([self.stock_data[:,self._idx,self.price_idx]])

        # value / weight evolution
        y_evol = prices_poststep/prices_prestep

        w_evol = (y_evol*w_refi)/(tf.matmul(y_evol,tf.transpose(w_refi)))

        # record the new portfolio value
        poststep_portfolio_value = refi_portfolio_value*(tf.matmul(y_evol,tf.transpose(w_refi)))
        self._portfolio_value = poststep_portfolio_value[0,0]
        self.portfolio_value_hist.append(self._portfolio_value)
        self._portfolio_weights = w_evol

        # calculate reward 
        reward = (tf.math.log(poststep_portfolio_value / prestep_portfolio_value))[0,0]
        info = []
        self.rewards_hist.append(reward)

        return self._get_observation(), reward, self._done, info

    def _get_observation(self):    
        '''
        Return window of normalized stock data and current econ data.
        '''

        window_stock_data = self.stock_data[:, self._idx-self.window_size+1:self._idx+1]
        #Normalize by timed prices:
        X = window_stock_data
        X = tf.transpose(X,(2,1,0))
        X = X / X[0,0,:]
        X = tf.transpose(X,(2,1,0))

        #econ_data = self.econ_data[self._idx]
        #econ_data = self.econ_data[0]
        return X#, econ_data