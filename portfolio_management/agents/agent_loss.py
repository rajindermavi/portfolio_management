import tensorflow as tf

import random

def agent_loss(env,agent,dsct = 1):
    total_loss = tf.convert_to_tensor(0.0)
    obs = env.reset()
    done = False 
    last_raw_action = tf.zeros((1,env.n_stocks+1))
    while not done:
        action,raw_action=agent.act(obs,last_raw_action)
        obs,reward,done,_=env.step(action)
        last_raw_action=raw_action
        total_loss *= dsct
        total_loss -= reward
    return total_loss  


TRADING_DAYS_PER_YEAR = 253

def sampled_agent_reward(Env,data,agents,n_group_size,trials,dsct = 1):

    n_stocks = data.shape[0]

    rewards = {agent.name:[] for agent in agents}

    for _ in range(trials):

        sample_idx = random.sample(list(range(n_stocks)),n_group_size)
        sample = data[sample_idx,:,:]
        
        env = Env(sample)    

        for agent in agents:

            loss = agent_loss(env,agent,dsct = dsct)     

            avg_gain = -TRADING_DAYS_PER_YEAR*loss/(env._end_tick-env._start_tick)  

            rewards[agent.name].append(avg_gain.numpy())

    return rewards  