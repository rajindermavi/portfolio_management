from environment import TradingEnv

import numpy as np

import tensorflow as tf

def test_enviroment():
    stocks_num = 10
    total_length = 100
    win_size = 50

    data = np.random.rand(stocks_num,total_length,3)

    env = TradingEnv(data,window_size=win_size)

    obs = env.reset()

    assert(obs.shape == (stocks_num+1,win_size,3))

    Ones = np.ones(stocks_num+1)
    action = tf.convert_to_tensor([Ones/Ones.sum()],dtype=tf.float32)

    obs, _, _, _ = env.step(action)

    assert(obs.shape == (stocks_num+1,win_size,3))
 
    


