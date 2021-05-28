import numpy as np
#import pandas as pd 
#from collections import deque

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
#from tensorflow.keras.layers import Dense, Dropout
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.regularizers import l2

class DPMAgent:
    '''
    constructs model which takes
    1. 3 dim tensor X ( ~ n_feats x n_stocks x window_size ) of stock data 
    2. 1 dim vector of econ data

    returns stock valuation position.
    '''
    def __init__(self,n_stocks,n_stock_feats,window_size,n_econ_feats):
        
        self.n_stocks = n_stocks
        self.n_stock_feats = n_stock_feats
        self.window_size = window_size
        self.n_econ_feats = n_econ_feats
        
        #
        self._w = np.zeros((1,10,1,1)) 

        self.model = self.build()

    def build(self):
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
        soft_max = layers.Softmax()(flatten) 

        model = keras.Model(inputs=input_layer, outputs=soft_max)
        return model  
 



