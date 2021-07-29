import tensorflow as tf

def agent_loss(env,agent,dsct = 1):
    total_loss = tf.convert_to_tensor(0.0)
    obs = env.reset()
    done = False 
    last_raw_action = tf.zeros((1,env.n_stocks+1))
    while not done:
        action,raw_action=agent.act(obs,last_raw_action)
        obs,reward,done,_=env.step(action)
        last_raw_action=raw_action
        total_loss-= dsct * reward
    return total_loss           