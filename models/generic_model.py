from keras.layers import Input
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Flatten
import keras.models as kd


class Model:
    def __init__(self, config):
        self.input_shape = config['Input_shape']
        self.layer_shapes = config['layer_shapes']
        self.activation = config['Activation']
        self.output = config['Output']
        self.model_type = config['model_type']
        self.model = self.construct_model(config)
        self.filter_size = []
        self.strides = []
        self.padding = []
        
    def construct_model(self, config):
        assert self.model_type in ['conv', 'fc', 'fully_connected']
        if self.model_type == 'conv':
            self.filter_size = config['filter_size']
            self.strides = config['stride']
            self.padding = config['padding']
            model = self.construct_conv_model()
        else:
            model = self.construct_fc_model()
        return model
            
    def construct_fc_model(self):
        input_layer = Input(shape=(self.input_shape,))
        layer = input_layer
        
        for layer_neurons in self.layer_shapes:
            layer = Dense(layer_neurons, activation=self.activation)(layer)
            
        layer = Dense(self.output)(layer)
        return kd.Model(input_layer, layer)
    
    def construct_conv_model(self):
        input_layer = Input(shape=self.input_shape)
        layer = input_layer
        for num_filters, filter_size, stride, padding in zip(self.layer_shapes,
                                                             self.filter_size, self.strides, self.padding):
            layer = Conv2D(num_filters, filter_size, strides=stride, padding=padding, activation=self.activation)(layer)
        layer = Flatten()(layer)
        layer = Dense(self.output,activation='softmax')(layer)
        return kd.Model(input_layer, layer)
            
        
                      
 

