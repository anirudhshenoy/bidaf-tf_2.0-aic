from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Activation, Multiply, Add, Lambda
from tensorflow.keras.initializers import Constant
import tensorflow as tf


class Highway(Layer):

    activation = None
    transform_gate_bias = None

    def __init__(self, activation='relu', transform_gate_bias=-1, **kwargs):

        self.activation = activation
        self.transform_gate_bias = transform_gate_bias
        super(Highway, self).__init__(**kwargs)
        

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.input_dim = input_shape[-1]
        
        self.W_H = self.add_weight(name = 'Highway_W_H',
                                   shape = (self.input_dim , self.input_dim),
                                   initializer = 'glorot_uniform',
                                   trainable = True)
        
        self.W_T = self.add_weight(name = 'Highway_W_T',
                                   shape = (self.input_dim , self.input_dim),
                                   initializer = 'glorot_uniform',
                                   trainable = True)
        
        self.b_H = self.add_weight(name = 'Highway_b_H',
                                   shape = (self.input_dim, ),
                                   initializer = 'zeros',
                                   trainable = True)
        
        self.b_T = self.add_weight(name = 'Highway_b_T',
                                   shape = (self.input_dim, ),
                                   initializer = Constant(self.transform_gate_bias),
                                   trainable = True)
        
        super(Highway, self).build(input_shape)  # Be sure to call this at the end
        

    def call(self, x):
        dim = K.int_shape(x)[-1]
        
        transform_gate = tf.matmul(x, self.W_T) + self.b_T
        transform_gate = Activation("sigmoid")(transform_gate)

        carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
        
        transformed_data = tf.matmul(x, self.W_H) + self.b_H
        transformed_data = Activation(self.activation)(transformed_data)
        
        transformed_gated = Multiply()([transform_gate, transformed_data])
        
        identity_gated = Multiply()([carry_gate, x])
        
        value = Add()([transformed_gated, identity_gated])
        
        return value

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config['activation'] = self.activation
        config['transform_gate_bias'] = self.transform_gate_bias
        return config