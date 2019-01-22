import numpy as np
import tensorflow as tf
import keras.layers as L
from keras.models import Model
from keras import backend as KB
import os
import sys

ROOTDIR=os.path.abspath("../../")
sys.path.append(ROOTDIR)

from utils.losses import weighted_cross_entropy_loss,f1_score_metric
from utils.callbacks import get_tensorboard_callback,get_checkpoint_call_back
from utils.tools import construct_model_from_csv
from models.BaseModel import Base_Model


class FC_CONV(Base_Model):
    
    def __init__(self,input_shape,num_outputs,optimizer,arch_config_files):
        """
        params:
            input_shape: a list that specifies input_shapes 
                            of all the inputs to the model (batch dim not included)
            num_outputs:num of neurons in the last layer
            optimizer: optimizer function used during training
            arch_config_files:list of 3 files that specify the design of conv network,fc network
                                and the network that combines output of both these networks
        """
        assert len(input_shape)==2
        Base_Model.__init__(self,input_shape,num_outputs,optimizer,"FC_CONV")
        assert len(arch_config_files)==3
        self.arch_config_files=arch_config_files
        
    def construct_model(self):
        """
        Model is constructed here and self.Model stores the link to the Model
        """
        conv_input=L.Input(shape=(self.input_shape[0]))
        scalar_features=L.Input(shape=(self.input_shape[1]))
        
        flattened_conv_features=construct_model_from_csv(self.arch_config_files[0],conv_input)
        fc_features=construct_model_from_csv(self.arch_config_files[1],scalar_features)
        concatenated_features=L.concatenate([flattened_conv_features,fc_features])
        logits=construct_model_from_csv(self.arch_config_files[2],concatenated_features)
        assert logits.shape[-1]==self.num_outputs
        self.Model=Model(inputs=[conv_input,scalar_features],output=logits)
        
        f1=f1_score_metric(self.num_outputs)        
        self.Model.compile(loss=weighted_cross_entropy_loss, optimizer=self.optimizer,metrics=["acc",f1])
        
        self.Model.summary()
        
        
        