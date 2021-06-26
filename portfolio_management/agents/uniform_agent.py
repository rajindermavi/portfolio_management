import numpy as np
import tensorflow as tf
class Uniform_Agent():
    def __init__(self,n_stocks):
        self.n_stocks = n_stocks

    def act(self,*args):

        Ones = np.ones(self.n_stocks+1)
        action = tf.convert_to_tensor([Ones/Ones.sum()],dtype=tf.float32)
        # Raw action is non-functional for this agent
        raw_action = tf.zeros((1,self.n_stocks+1))
        return action, raw_action