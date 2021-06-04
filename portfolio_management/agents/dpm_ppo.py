
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import numpy as np
import tensorflow_probability as tfp

class actor_critic_model(tf.keras.Model):
    def __init__(self,n_stocks,n_stock_feats,role,window_size=64):
        super().__init__(self)
        assert role in ['actor','critic'], print('Expected fourth argument "role" to be either "actor" or "critic".')

        self.n_stocks = n_stocks
        self.n_stock_feats = n_stock_feats
        self.window_size = window_size 

        self.cash_bias = 0.5 * np.ones((1,1,1)) 
        self._w = np.zeros((n_stocks,1,1)) 

        self.build_layers(role)

    def build_layers(self,role):
        #input_shape = (self.n_stocks,self.window_size,self.n_stock_feats)
        #self.input_layer = layers.Input(shape = input_shape)
        filters_1=2
        kernel_size_1 = (1,3)
        self.conv_layer_1 = layers.Conv2D(filters_1,
                                     kernel_size_1,
                                     name='conv1',
                                     padding='valid',
                                     activation='relu')
        filters_2 = 20
        kernel_size_2 = (1,self.window_size-2)
        self.conv_layer_2 = layers.Conv2D(filters_2,
                                     kernel_size_2,
                                     name='conv2',
                                     padding='valid',
                                     activation='relu')
         
        #w_ = tf.expand_dims(tf.convert_to_tensor(w),axis =0)
        self.concat_layer_1 = layers.Concatenate(axis=3)
        filters_3 = 1
        kernel_size_3 = (1,1)
        self.conv_layer_3 = layers.Conv2D(filters_3,
                                     kernel_size_3,
                                     padding='valid',
                                     activation='relu') 
        
        self.concat_layer_2 = layers.Concatenate(axis=1)#([cash_bias, conv_layer_3])
        self.flatten = layers.Flatten()

        if role == 'critic':
            self.output_layer = layers.Dense(1,activation = 'tanh',name = 'predictions')


        if role == 'actor':
            self.output_layer = layers.Softmax()

    def call(self,input_data):
        batch_size = input_data.shape[0]
        w = np.array(batch_size*[self._w])
        cash_bias = np.array(batch_size*[self.cash_bias])
        #x = self.input_layer(input_data)
        x = self.conv_layer_1(input_data)
        x = self.conv_layer_2(x)
        x = self.concat_layer_1([x,w])
        x = self.conv_layer_3(x)
        x = self.concat_layer_2([cash_bias,x])
        x = self.flatten(x)
        return self.output_layer(x)
        



class Agent():
    def __init__(self,n_stocks,n_stock_feats,window_size=64, gamma = 0.99):
        self.n_portfolio = n_stocks + 1
        self.clip_pram = 0.2
        self.gamma = gamma
        self.a_opt = optimizers.RMSprop(learning_rate=7e-3)
        self.c_opt = optimizers.RMSprop(learning_rate=7e-3)
        self.actor = actor_critic_model(n_stocks,n_stock_feats,'actor',window_size=window_size)
        self.critic = actor_critic_model(n_stocks,n_stock_feats,'critic',window_size=window_size)

          
    def act(self,state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        #dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        #action = dist.sample()
        #return int(action.numpy()[0])    
        return prob 

    def actor_loss(self, probs, actions, adv, old_probs, closs):
        
        probability = probs      
        entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(probability,tf.math.log(probability))))
        #print(probability)
        #print(entropy)
        sur1 = []
        sur2 = []
        
        for pb, t, op in zip(probability, adv, old_probs):
                        t =  tf.constant(t)
                        op =  tf.constant(op)
                        #print(f"t{t}")
                        #ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        ratio = tf.math.divide(pb,op)
                        #print(f"ratio{ratio}")
                        s1 = tf.math.multiply(ratio,t)
                        #print(f"s1{s1}")
                        s2 =  tf.math.multiply(tf.clip_by_value(ratio, 1.0 - self.clip_pram, 1.0 + self.clip_pram),t)
                        #print(f"s2{s2}")
                        sur1.append(s1)
                        sur2.append(s2)

        sr1 = tf.stack(sur1)
        sr2 = tf.stack(sur2)
        
        #closs = tf.reduce_mean(tf.math.square(td))
        loss = tf.math.negative(tf.reduce_mean(tf.math.minimum(sr1, sr2)) - closs + 0.001 * entropy)
        #print(loss)
        return loss

    def learn(self, states, actions,  adv , old_probs, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        adv = tf.reshape(adv, (len(adv),))

        old_p = old_probs

        old_p = tf.reshape(old_p, (len(old_p),self.n_portfolio))
        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v =  self.critic(states,training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            c_loss = 0.5 * losses.mean_squared_error(discnt_rewards, v)
            a_loss = self.actor_loss(p, actions, adv, old_probs, c_loss)
            
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss



def preprocess(states, actions, rewards, dones, values, gamma):
    g = 0
    lmbda = 0.95
    returns = []
    for i in reversed(range(len(rewards))):
       delta = rewards[i] + gamma * values[i + 1] * dones[i] - values[i]
       g = delta + gamma * lmbda * dones[i] * g
       returns.append(g + values[i])

    returns.reverse()
    adv = np.array(returns, dtype=np.float32) - values[:-1]
    adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    returns = np.array(returns, dtype=np.float32)
    return states, actions, returns, adv            