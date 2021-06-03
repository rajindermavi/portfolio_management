import numpy as np
#import pandas as pd 
#from collections import deque

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
#from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.regularizers import l2
import keras.backend as K


def ppo_loss(oldpolicy, advantages, rewards, values):
    def loss(y_true, y_pred):
        critic_discount = 0.5
        clipping_val = 0.2
        entropy_beta = 0.001
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

class DPMAgent:
    '''
    constructs model which takes
    1. 3 dim tensor X ( ~ n_feats x n_stocks x window_size ) of stock data 
    2. 1 dim vector of econ data

    returns stock valuation position.
    '''
    def __init__(self,n_stocks,n_stock_feats,n_econ_feats,role,window_size=64):
        assert role in ['actor','critic'], print('Expected fourth argument "role" to be either "actor" or "critic".')

        self.n_stocks = n_stocks
        self.n_stock_feats = n_stock_feats
        self.window_size = window_size
        self.n_econ_feats = n_econ_feats
        
        self._w = np.zeros((1,n_stocks,1,1)) 

        self.model = self.build(role)

    def build(self,role):

        input_shape = (self.n_stocks,self.window_size,self.n_stock_feats)
        input_layer = layers.Input(shape = input_shape)
        filters_1=2
        kernel_size_1 = (1,3)
        conv_layer_1 = layers.Conv2D(filters_1,
                                     kernel_size_1,
                                     name='conv1',
                                     padding='valid',
                                     activation='relu',
                                     input_shape=input_shape)(input_layer)
        filters_2 = 20
        kernel_size_2 = (1,self.window_size-2)
        conv_layer_2 = layers.Conv2D(filters_2,
                                     kernel_size_2,
                                     name='conv2',
                                     padding='valid',
                                     activation='relu')(conv_layer_1)
        w = self._w
        #w_ = tf.expand_dims(tf.convert_to_tensor(w),axis =0)
        concat_layer_1 = layers.Concatenate(axis=3)([conv_layer_2,w])
        filters_3 = 1
        kernel_size_3 = (1,1)
        conv_layer_3 = layers.Conv2D(filters_3,
                                     kernel_size_3,
                                     padding='valid',
                                     activation='relu')(concat_layer_1)
        cash_bias = 0.5 * np.ones((1,1,1,1)) 
        concat_layer_2 = layers.Concatenate(axis=1)([cash_bias, conv_layer_3])
        flatten = layers.Flatten()(concat_layer_2)

        if role == 'critic':
            output_layer = layers.Dense(1,activation = 'tanh',name = 'predictions')(flatten)
            model = keras.Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(lr=1e-4), loss='mse')

        if role == 'actor':
        
            
            soft_max = layers.Softmax()(flatten) 

            output_dims = soft_max.shape
            oldpolicy = layers.Input(shape=output_dims)
            advantages = layers.Input(shape=(1, 1,))
            rewards = layers.Input(shape=(1, 1,))
            values = layers.Input(shape=(1, 1,))

            model = keras.Model(inputs=[input_layer,
                                        oldpolicy,
                                        advantages,
                                        rewards,
                                        values], outputs=[soft_max])
            
      
            model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
                oldpolicy=oldpolicy,
                advantages=advantages,
                rewards=rewards,
                values=values)])
        return model  
 
    def update_weights(self, weights):
        '''
        Update weights of the portfolio.
        '''
        self._w = weights