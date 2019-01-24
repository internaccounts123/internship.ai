from DeepNN import DNN
import numpy as np
import os
import pickle


class Network:
    def __init__(self, path):
        self.path = path
        self.config = {'layer_shapes': [500, 500, 500], 'Activation': 'relu',
                       'Output': 6, 'Input_shape': 472,
                       'Weights': os.path.join(self.path, 'Weights.h5'), 'dropout': 0.2}
        self.dnn = DNN(self.config)

    def predict(self, example):
        if example[-2] == 'Unknown':
            example[-2] = 'Accelerate'

        with open(os.path.join(self.path, "lane_change_mode.pkl"), 'rb') as f:
            lane_encoder = pickle.load(f)
        with open(os.path.join(self.path, "previous_decision.pkl"), 'rb') as f:
            decision_encoder = pickle.load(f)
        with open(os.path.join(self.path, "action.pkl"), 'rb') as f:
            action_encoder = pickle.load(f)
            
        example[-2] = decision_encoder.transform([example[-2]])[0]
        example[1] = lane_encoder.transform([example[1]])[0]
        example = example.astype(float)
        action = np.argmax(np.array(self.dnn.predict(example)), axis=1)[0]
        return action_encoder.inverse_transform([action])[0]
