from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Activation
from keras.models import Model
import tensorflow as tf
from keras.losses import categorical_crossentropy
import numpy as np
from keras.optimizers import Adam


class DNN:
    """
    Class to construct Fully Connected Deep Networks
    """
    def __init__(self, config):
        """

        :param config: Dictionary of network configurations
        """
        self.input_shape = config['Input_shape']
        self.layer_shapes = config['layer_shapes']
        self.activation = config['Activation']
        self.output = config['Output']
        self.weights = config['Weights']
        self.config = config
        self.model = self.construct_fc_model()
        self.compile()
        if self.weights:
            self.load_weights()

    def construct_fc_model(self):
        """
        Constructs a Fully Connected model
        :return: Model object of keras
        """
        input_layer = Input(shape=(self.input_shape,))
        layer = input_layer
        layer = BatchNormalization()(layer)
        for layer_neurons in self.layer_shapes:
            layer = Dense(layer_neurons)(layer)
            layer=Activation('relu')(layer)
            layer = BatchNormalization()(layer)
            layer = Dropout(self.config['dropout'])(layer)
        layer = Dense(self.output)(layer)
        layer=Activation(tf.nn.softmax)(layer)
        return Model(input_layer, layer)
      
    def load_weights(self):
        """
        Load weights into a model
        :return: Nothing
        """
        self.model.load_weights(self.weights)

    def predict(self, input):
        """
        Returns the predicted label
        """
        return self.model.predict(np.atleast_2d(input))

    def compile(self):
        self.model.compile(loss=categorical_crossentropy, optimizer=Adam(0.001, decay=1e-4), metrics=['acc'])
