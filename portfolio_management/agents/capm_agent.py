import numpy as np
import tensorflow as tf

from scipy.optimize import LinearConstraint
from scipy.optimize import minimize


class CAPM_Agent():
    def __init__(self,n_stocks,risk_free_rate = 1e-4):

        self.n_stocks = n_stocks
        self.r = risk_free_rate

    def act(self,*args):
        obs = args[0]

        K = ((obs[:-1,1:,-1] - obs[:-1,:-1,-1])/obs[:-1,:-1,-1])
        self.Sigma = np.cov(K)
        self.mu = np.mean(K,axis=1)

        Ones = np.ones(self.n_stocks)
        I = np.eye(self.n_stocks)
        eq = LinearConstraint(Ones, [1], [1])
        ineq = LinearConstraint(I,self.n_stocks*[0],self.n_stocks*[1])
        w0 = Ones / Ones.sum()

        res = minimize(self.market_portfolio_obj_fun,
                        w0, constraints=[eq,ineq])
        action = np.concatenate([res['x'],[0]])
        action = tf.convert_to_tensor([action],dtype=tf.float32)
        # Raw action is non-functional for this agent
        raw_action = tf.zeros((1,self.n_stocks+1))

        return action, raw_action

        return 

    def market_portfolio_obj_fun(self,w):
        num = np.dot(w,self.mu) - self.r
        denom =np.sqrt(np.matmul(np.matmul(w,self.Sigma),w))
        return -num/denom
