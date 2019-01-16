import tensorflow as tf
import keras.layers as L
from utils.callbacks import get_tensorboard_callback,get_checkpoint_call_back



class Base_Model():
    def __init__(self,input_shape,num_outputs,optimizer,prefix):
        """
        params:
            input_shape: a list that specifies input_shapes 
                            of all the inputs to the model (batch dim not included)
            num_outputs:num of neurons in the last layer
            optimizer: optimizer function used during training
            prefix: name of the model (this name is appended to the checkpoints and logs)
                        
        """
        
        self.input_shape=input_shape
        self.num_outputs=num_outputs
        self.optimizer=optimizer
        self.prefix=prefix
        
    def construct_model(self):
        """
        Model is constructed here and self.Model stores the link to the Model
        """
        raise Exception('Unimplemented Error ')
    
        
    def train(self,generator,logdir="../../logs",checkpoint_dir="../../checkpoints",save_N_epochs=10):
        """
        
        """
        tensorboard=get_tensorboard_callback(logdir=logdir+"/"+self.prefix)
        checkpoint=get_checkpoint_call_back(checkpoint_dir,self.prefix+"_chkpoint_",period=save_N_epochs)
        
        self.Model.fit_generator(generator= generator, 
                    steps_per_epoch  = len(generator), 
                    epochs           = self.config['epochs'], 
                    verbose          = 1,
                    max_queue_size   = 3,
                                callbacks=[tensorboard,checkpoint])
        
    def predict(self,observation):
        """
        predicts output given an input
        """
        
        raise Exception('Unimplemented Error ')
        