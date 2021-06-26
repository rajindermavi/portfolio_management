import tensorflow as tf

def agent_loss(env,agent,n_stocks):
    total_loss = tf.convert_to_tensor(0.0)
    obs = env.reset()
    done = False 
    last_raw_action = tf.zeros((1,n_stocks+1))
    while not done:
        action,raw_action=agent.act(obs,last_raw_action)
        obs,reward,done,_=env.step(action)
        last_raw_action=raw_action
        total_loss-=reward
    return total_loss           