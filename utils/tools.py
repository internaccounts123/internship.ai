import pandas as pd
import keras.layers as layers


def construct_model_from_csv(model_file, input_placeholder):
        """

        :param model_file: path of csv file containing info about the design of the model
        :param input_placeholder: this is used as input to the model
        :return x: the last layer (tensor) of the model
        """
        model_df = pd.read_csv(model_file).fillna(0)
        model_df = model_df.astype({"layer_type": str, "units": int, "strides":int,
                                   "padding": str, "kernel_size": int, "num_filters": int})
        x=input_placeholder
        for i in range(model_df.shape[0]):
            layer_type=model_df.layer_type[i]
            assert layer_type in ["conv", "conv1D", "fc", "flatten", "relu","elu", "bn", "pooling"],\
                "invalid layer type"
            
            if layer_type == "conv":
                padding=str(model_df.padding[i])
                kernel_size = int(model_df.kernel_size[i])
                num_filters = int(model_df.num_filters[i])
                strides = int(model_df.strides[i])
                
                assert padding == "same" or padding == "valid"
                assert kernel_size > 0
                assert num_filters > 0
                assert strides > 0

                x = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)

            if layer_type == "conv1D":
                padding = str(model_df.padding[i])
                kernel_size = int(model_df.kernel_size[i])
                num_filters = int(model_df.num_filters[i])
                strides = int(model_df.strides[i])

                assert padding == "same" or padding == "valid"
                assert kernel_size > 0
                assert num_filters > 0
                assert strides > 0

                x = layers.Conv1D(num_filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
                
            elif layer_type == "fc":
                hidden_units = int(model_df.units[i])
                assert hidden_units > 0
                x = layers.Dense(hidden_units)(x)
                
            elif layer_type == "flatten":
                x = layers.Flatten()(x)
                
            elif layer_type == "relu":
                x = layers.Activation("relu")(x)
                
            elif layer_type == "elu":
                x = layers.Activation("elu")(x)
                
            elif layer_type == "bn":
                x = layers.BatchNormalization()(x)
                
            elif layer_type == "pooling":
                padding = int(model_df.padding[i])
                kernel_size = int(model_df.kernel_size[i])
                strides = int(model_df.strides[i])
                
                assert padding == "same" or padding == "valid"
                assert kernel_size > 0
                assert strides > 0
                x = layers.MaxPooling2D(kernel_size, strides=strides, padding=padding)(x)
                
        return x
