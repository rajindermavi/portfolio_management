from .uniform_agent import Uniform_Agent
from .capm_agent import CAPM_Agent
from .mvp_agent import MVP_Agent
from .dpm_agent import DPM_Agent

import tensorflow as tf
import numpy as np

def test_agents():

    agents = [Uniform_Agent(),
              CAPM_Agent(),
              MVP_Agent(),
              DPM_Agent()]

    for agent in agents:
        stocks_num = 10
        total_length = 50


        data = np.random.rand(stocks_num,total_length,5)
        last_action = tf.convert_to_tensor(np.random.rand(1,stocks_num))

        action, _ = agent.act(data,last_action)

        assert( action.shape == (1,stocks_num)) 

