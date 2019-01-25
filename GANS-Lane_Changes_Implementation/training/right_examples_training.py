

"""
	@Author: Salman Ahmed

"""





import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam


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


def build_discriminator(img_rows=100, img_cols=3, channels=1):
    img_shape = (img_rows, img_cols, channels)
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    img = Input(shape=img_shape)
    validity = model(img)
    return Model(img, validity)


data = pd.read_csv("data/right_lane.csv")
data.pop('action')
data = data.sample(frac=1)
ob_netCols = []
k = 0
while k < 300:
    ob_netCols.append("new_col"+str(k))
    k += 1
ob_net = data[ob_netCols]
ob_net = np.reshape(ob_net.values, (ob_net.shape[0],100,3,1))
ob_net[ob_net>0] = 0
ob_net[ob_net==-1] = -5
ob_net = ob_net + 10
ob_net = ob_net / 10
ob_net = ob_net * 2
ob_net = ob_net - 1
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])
generator.compile(loss='binary_crossentropy',optimizer=Adam(0.0002, 0.5))
overallModel = Sequential()
overallModel.add(generator)
discriminator.trainable = False
overallModel.add(discriminator)
overallModel.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
BATCH_SIZE = int(ob_net.shape[0]/3)
indc = []
j = 0
while j < ob_net.shape[0]:
    indc.append(j)
    j+=1
for epoch in range(1000):
    noise = np.random.uniform(0, 1.0, size=(BATCH_SIZE,100))
    fakehR = generator.predict(noise)
    valid = np.ones((BATCH_SIZE,1))
    fake = np.zeros((BATCH_SIZE,1))
#       valid[8] = [[[0]]]
#       valid[13] = [[[0]]]


    valid = valid * (1 - 0.1)
    fake = fake + (0.1)

    # highR = data.sample(n=BATCH_SIZE)
    index = np.random.choice(indc, BATCH_SIZE, replace=False)  
    highR = ob_net[index]

    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(highR, valid)
    d_loss_fake = discriminator.train_on_batch(fakehR, fake)
    k = 0
    d_loss = (d_loss_real[0] + d_loss_fake[0]) * 0.5


    y = np.ones((BATCH_SIZE,1))
    discriminator.trainable = False
    j = 0
    while j < 5:
        noise = np.random.uniform(0, 1.0, size=(BATCH_SIZE,100))
        
        g_loss = overallModel.train_on_batch(noise,y)
        print ("Epoch : ",epoch, "  Discriminator Loss : ", d_loss, "  Generator Loss : ", g_loss)
        j+=1
generator.save_weights("Right_GEN_WEIGHTS.h5")

