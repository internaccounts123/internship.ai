import pandas as pd
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
import gc
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, Activation, Flatten, concatenate, Input
from sklearn.utils import class_weight


"""
	Author: Salman Ahmed

"""



def get_data(no_file=1):
    paths = os.listdir(os.path.curdir)
    data = pd.DataFrame()
    file_number = 1
    for each in paths:
        if data.shape[0] >= no_file:
            break
        n_path = os.path.join(os.path.curdir, each)
        temp = pd.read_hdf(n_path)
        temp.pop('id')
        temp.pop('profile')
        temp.pop('observe_net_shape')
        temp.pop('key')
        temp.speed = temp.speed*3.6
        temp.speed_limit = temp.speed_limit*3.6
        temp = temp[temp.action != 'AbortLaneChange']
        temp1 = temp[temp.action == 'LeftLaneChange']
        temp2 = temp[temp.action == 'RightLaneChange']
        temp = temp[temp.action != 'LeftLaneChange']
        temp = temp[temp.action != 'RightLaneChange']
        lane_change = temp1.shape[0] + temp2.shape[0]
        if temp.shape[0] >= (4*lane_change):
            temp = temp.sample(n=4*lane_change)
        temp = pd.concat([temp, temp1, temp2])
        print (temp.shape, data.shape, file_number)
        if temp.shape[0] > 0:
            temp.pop('driver_type')
            data = pd.concat([data, temp])
        file_number += 1
    return data


data = get_data(800000)


def get_model_cnn():
    inp1 = Input(shape=(155, 3, 1))
    temp = Conv2D(512, (1, 1))(inp1)
    temp = Activation('relu')(temp)
    temp = Conv2D(256, (1, 1))(temp)
    temp = Activation('relu')(temp)
    temp = Conv2D(128, (1, 1))(temp)
    temp = Activation('relu')(temp)
    temp = Conv2D(64, (1, 1))(temp)
    temp = Activation('relu')(temp)
    temp = Flatten()(temp)
    temp = Dense(32, activation='relu')(temp)

    inp2 = Input(shape=(10,))
    temp1 = Dense(512, activation='relu')(inp2)
    temp1 = Dense(256, activation='relu')(temp1)
    temp1 = Dense(128, activation='relu')(temp1)
    temp1 = Dense(64, activation='relu')(temp1)
    temp1 = Dense(32, activation='relu')(temp1)

    merge = concatenate([temp, temp1])

    temp = Dense(12, activation='relu', name="get_feature_map")(merge)
    temp = Dense(5)(temp)
    temp = Activation('sigmoid')(temp)
    model = Model([inp1, inp2], temp)
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


y = data.pop('action')
class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
ob_netCols = []
k = 0
while k < 465:
    ob_netCols.append("new_col"+str(k))
    k += 1
ob_net = data[ob_netCols]
for each in ob_netCols:
    data.pop(each)
y[y == 'Accelerate'] = 0
y[y == 'Decelerate'] = 1
y[y == 'Keep'] = 2
y[y == 'LeftLaneChange'] = 3
y[y == 'RightLaneChange'] = 4
y = y.values
y = to_categorical(y)
ob_net = np.reshape(ob_net.values, (ob_net.shape[0], 155, 3, 1))
data.previous_decision[data.previous_decision == 'Accelerate'] = 0
data.previous_decision[data.previous_decision == 'Decelerate'] = 1
data.previous_decision[data.previous_decision == 'Keep'] = 2
data.previous_decision[data.previous_decision == 'LeftLaneChange'] = 3
data.previous_decision[data.previous_decision == 'RightLaneChange'] = 4
data['Acc'] = data.previous_decision == 0
data['Dec'] = data.previous_decision == 1
data['Keep'] = data.previous_decision == 2
data['Left'] = data.previous_decision == 3
data['Right'] = data.previous_decision == 4
data.pop('previous_decision')
data[data.lane_change_mode == 'DRIVE_MODE_AUTO'] = 0
data[data.lane_change_mode == 'DRIVE_MODE_CHANGELANE'] = 1
model = get_model_cnn()
model.summary()
model.fit([ob_net, data], y, epochs=20, class_weight=class_weights)
model.save_weights('FCNN_CNN.h5')

