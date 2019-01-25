import numpy as np 
import pandas as pd
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam 
import os


def build_generator(noise_shape = (100,), img_shape=(100,3,1)):
    model = Sequential()
    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    noise = Input(shape=noise_shape)
    img = model(noise)
    return Model(noise, img)


def generate_left_example(generator):
    noise = np.random.uniform(0, 1.0, size=(1,100))
    pred = generator.predict(noise)
    temp = pred < 0.35
    temp1 = pred > -0.35
    temp = temp * temp1
    pred[temp] = -1
    temp = pred >= 0.35
    pred[temp] = 0
    temp = pred < -0.35
    temp1 = pred > -0.999999999
    temp = temp * temp1
    pred[temp] = -10
    pred = np.reshape(pred, (1,100,3))
    return pred

    
def generate_plot_left(generator):
    %pylab inline
    example = generate_left_example(generator)
    example = example.astype('int32')
    example[example == -1] = -100
    example[example == 0] = 0
    example[example == -10] = 100
    fig = plt.figure()
    fig.set_size_inches(5,12)
    plt.imshow(example[0], cmap='binary')



model_path = "../gans_weights/left_lane_weights/GEN_WEIGHTS.h5"
generator = build_generator()
generator.compile(loss='binary_crossentropy',optimizer=Adam(0.0002, 0.5))
generator.load_weights(model_path)
generate_plot_left(generator)