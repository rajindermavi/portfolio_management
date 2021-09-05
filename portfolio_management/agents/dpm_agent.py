
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
import numpy as np

class Model(tf.keras.Model):
    def __init__(self,window_size=50):
        super().__init__(self)
         
        self.window_size = window_size 

        self.cash_bias = 0.5 * np.ones((1,1,1)) 

        self.build_layers()

    def build_layers(self): 

        filters_1 =4
        conv_len = 2
        kernel_size_1 = (1,1 + 2*conv_len)
        self.conv_layer_1 = layers.Conv2D(filters_1,
                                     kernel_size_1,
                                     name='conv1',
                                     padding='valid',
                                     activation='linear',
                                     kernel_regularizer=regularizers.L2(l2 = 1e-2),
                                     bias_regularizer=regularizers.L2(l2 = 1e-2),
                                     kernel_initializer = 'he_normal')
        self.activation_layer_1 = layers.LeakyReLU(name='act1')
        self.batch_norm_layer_1 = layers.BatchNormalization(name='bn1')

        filters_2 = 30
        kernel_size_2 = (1,self.window_size-2*conv_len)
        self.conv_layer_2 = layers.Conv2D(filters_2,
                                     kernel_size_2,
                                     name='conv2',
                                     padding='valid',
                                     activation='linear',
                                     kernel_regularizer=regularizers.L2(l2 = 1e-1),
                                     bias_regularizer=regularizers.L2(l2 = 1e-1),
                                     kernel_initializer = 'he_normal')
        self.activation_layer_2 = layers.LeakyReLU(name='act2')
        self.batch_norm_layer_2 = layers.BatchNormalization(name='bn2')
        
        filters_3 = 1
        kernel_size_3 = (1,1)
        self.conv_layer_3 = layers.Conv2D(filters_3,
                                     kernel_size_3,
                                     name = 'conv3',
                                     padding='valid',
                                     activation='linear',
                                     kernel_regularizer=regularizers.L2(l2 = 1e-1),
                                     bias_regularizer=regularizers.L2(l2 = 1e-1),
                                     kernel_initializer = 'he_normal') 

        
        self.flatten = layers.Flatten()

        self.weighted_vec1 = ScaleLayer()
        self.weighted_vec2 = ScaleLayer()
        self.average_layer = layers.Average()
        self.softmax_layer = layers.Softmax() 

    def call(self,*args):
        ''' ''' 
        input_data = args[0]
        last_raw_action = args[1]
        x = self.conv_layer_1(input_data)
        x = self.activation_layer_1(x)
        x = self.batch_norm_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.activation_layer_2(x)
        x = self.batch_norm_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.flatten(x)
        y1 = self.weighted_vec1(x)
        y2 = self.weighted_vec2(last_raw_action)
        x = self.average_layer([y1,y2])
        return x
               
class ScaleLayer(layers.Layer):
    def __init__(self):
        super(ScaleLayer, self).__init__()
        self.scale = tf.Variable(1.)

    def call(self, inputs):
        return inputs * self.scale
 
class DPM_Agent():
    name = "DPM_Agent"
    def __init__(self,window_size=50):


        self.ls = optimizers.schedules.PolynomialDecay(1e-2,1000,
                                                       end_learning_rate=1e-4)
        self.opt = optimizers.Adam(learning_rate=self.ls,clipnorm=1.0) 
        self.model = Model(window_size=window_size) 

    def act(self,*args):
        obs = args[0]
        last_action = args[1]
        raw_action = self.model(tf.convert_to_tensor([obs]),last_action)
        action = self.model.softmax_layer(raw_action)
        return action, raw_action