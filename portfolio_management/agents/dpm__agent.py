
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import numpy as np
import tensorflow_probability as tfp

class Model(tf.keras.Model):
    def __init__(self,n_stocks,n_stock_feats,window_size=64,debug=False):
        super().__init__(self)
        
        self.debug=debug
        self.n_stocks = n_stocks
        self.n_stock_feats = n_stock_feats
        self.window_size = window_size 

        self.cash_bias = 0.5 * np.ones((1,1,1)) 
        self._w = np.zeros((n_stocks,1,1)) 

        self.build_layers()

    def build_layers(self):
        #input_shape = (self.n_stocks,self.window_size,self.n_stock_feats)
        #self.input_layer = layers.Input(shape = input_shape)
        filters_1=2
        kernel_size_1 = (1,3)
        self.conv_layer_1 = layers.Conv2D(filters_1,
                                     kernel_size_1,
                                     name='conv1',
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer = 'glorot_normal')
        filters_2 = 20
        kernel_size_2 = (1,self.window_size-2)
        self.conv_layer_2 = layers.Conv2D(filters_2,
                                     kernel_size_2,
                                     name='conv2',
                                     padding='valid',
                                     activation='relu',
                                     kernel_initializer = 'glorot_normal')
         
        #w_ = tf.expand_dims(tf.convert_to_tensor(w),axis =0)
        self.concat_layer_1 = layers.Concatenate(axis=3,
                                                 name='conc1')
        filters_3 = 1
        kernel_size_3 = (1,1)
        self.conv_layer_3 = layers.Conv2D(filters_3,
                                     kernel_size_3,
                                     name = 'conv3',
                                     padding='valid',
                                     activation='relu') 
        
        #self.concat_layer_2 = layers.Concatenate(axis=1,
        #                                         name='conc2')
        self.flatten = layers.Flatten()

        self.weighted_vec1 = ScaleLayer()
        self.weighted_vec2 = ScaleLayer()
        self.average_layer = layers.Average()
        self.softmax_layer = layers.Softmax()


    def call(self,input_data,last_action):
        batch_size = input_data.shape[0]
        w = np.array(batch_size*[self._w])
        cash_bias = np.array(batch_size*[self.cash_bias])
        #x = self.input_layer(input_data)
        x = self.conv_layer_1(input_data)
        x = self.conv_layer_2(x)
        #x = self.concat_layer_1([x,w])
        if self.debug:
            print('first concat')
            print(x)
        x = self.conv_layer_3(x)
        x = self.flatten(x)
        y1 = self.weighted_vec1(x)
        y2 = self.weighted_vec2(last_action)
        x = self.average_layer([y1,y2])
        #x = tf.Variable(self.concat_layer_2([cash_bias,x]))

        return self.softmax_layer(x)
        
class ScaleLayer(layers.Layer):
    def __init__(self):
      super(ScaleLayer, self).__init__()
      self.scale = tf.Variable(1.)

    def call(self, inputs):
      return inputs * self.scale


class Agent():
    def __init__(self,n_stocks,n_stock_feats,window_size=64, gamma = 0.99):
        #self.n_portfolio = n_stocks + 1
        #self.clip_pram = 0.2
        #self.gamma = gamma
        self.opt = optimizers.RMSprop(learning_rate=1e-2) 
        self.model = Model(n_stocks,n_stock_feats,window_size=window_size) 

          
    def act(self,obs,last_action):
        action = self.model(tf.convert_to_tensor([obs]),last_action)
        #action = action.numpy() 
        return action
 


    def actor_loss(self, weights, adv, old_probs, closs):
        
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
                        ratio = tf.math.exp(tf.math.log(pb + 1e-10) - tf.math.log(op + 1e-10))
                        #ratio = tf.math.divide(pb,op)
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

    def learn(self,states):
        #print(states.shape)
        #print(adv.shape)
        #print(old_probs.shape)
        #print(discnt_rewards.shape)

        #discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))
        #adv = tf.reshape(adv, (len(adv),))
        #print(discnt_rewards.shape)
 
        with tf.GradientTape() as tape:
            weight_vectors = self.actor(states, training=True) 
            loss = self.actor_loss(weight_vectors)
            
        grads1 = tape.gradient(loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grads1, self.actor.trainable_variables))

        return loss

def batch_training_data(stock_data,batch_size):

    batches = []


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

def test_reward(agent,env):
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.actor(np.array([state])).numpy()
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
    return total_reward