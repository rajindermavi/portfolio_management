import numpy as np
import tensorflow as tf
class Uniform_Agent():
    name = "Uniform_Agent"
    def __init__(self):
        self.n_stocks = None

    def act(self,*args):

        obs = args[0]

        self.n_stocks = obs.shape[0] - 1

        Ones = np.ones(self.n_stocks+1)
        action = tf.convert_to_tensor([Ones/Ones.sum()],dtype=tf.float32)
        # Raw action is non-functional for this agent
        raw_action = tf.zeros((1,self.n_stocks+1))
        return action, raw_action