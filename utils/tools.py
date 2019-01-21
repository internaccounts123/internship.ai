import pandas as pd
import keras.layers as L

def construct_model_from_csv(model_file,input_placeholder):
        """
        @Author: Armughan
        given a csv file this function constructs a graph from it
        
        model_file: path of csv file that contains details about the model
        input placeholder: input tensor
        """
        model_df=pd.read_csv(model_file).fillna(0)
        model_df = model_df.astype({"layer_type": str, "units": int,"strides":int,
                                   "padding":str,"kernel_size":int,"num_filters":int})
        x=input_placeholder
        for i in range(model_df.shape[0]):
            layer_type=model_df.layer_type[i]
            assert layer_type in ["conv","conv1D","fc","flatten","relu","bn","pooling"],"invalid layer type"
            
            if layer_type=="conv":
                padding=str(model_df.padding[i])
                kernel_size=int(model_df.kernel_size[i])
                num_filters=int(model_df.num_filters[i])
                strides=int(model_df.strides[i])
                
                assert padding=="same" or padding=="valid"
                assert kernel_size>0
                assert num_filters>0
                assert strides>0
                
                x=L.Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding=padding)(x)

            if layer_type == "conv1D":
                padding = str(model_df.padding[i])
                kernel_size = int(model_df.kernel_size[i])
                num_filters = int(model_df.num_filters[i])
                strides = int(model_df.strides[i])

                assert padding == "same" or padding == "valid"
                assert kernel_size > 0
                assert num_filters > 0
                assert strides > 0

                x = L.Conv1D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
                
            elif layer_type=="fc":
                hidden_units=int(model_df.units[i])
                assert hidden_units>0
                x=L.Dense(hidden_units)(x)
                
            elif layer_type=="flatten":
                x=L.Flatten()(x)
                
            elif layer_type=="relu":
                x=L.Activation("relu")(x)
                
            elif layer_type=="bn":
                x=L.BatchNormalization()(x)
                
            elif layer_type=="pooling":
                padding=int(model_df.padding[i])
                kernel_size=int(model_df.kernel_size[i])
                strides=int(model_df.strides[i])
                
                assert padding=="same" or padding=="valid"
                assert kernel_size>0
                assert strides>0
                x=L.MaxPooling2D(kernel_size,strides=strides,padding=padding)(x)
                
        return x