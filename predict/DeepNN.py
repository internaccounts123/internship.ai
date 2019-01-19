from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Input
from keras.models import Model
import numpy as np


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
        self.model = self.construct_fc_model()
        if self.weights:
            self.load_weights()

    def construct_fc_model(self):
        """
        Constructs a Fully Connected model
        :return: Model object of keras
        """
        input_layer = Input(shape=(self.input_shape,))
        layer = input_layer
        for layer_neurons in self.layer_shapes:
            layer = Dense(layer_neurons, activation=self.activation)(layer)
        layer = Dense(self.output)(layer)
        return Model(input_layer, layer)

    def load_weights(self):
        """
        Load weights into a model
        :return: Nothing
        """
        self.model.load_weights(self.weights)

    def predict(self, input_example):
        """
        Returns the predicted label
        """
        return self.model.predict(np.atleast_2d(input_example))
