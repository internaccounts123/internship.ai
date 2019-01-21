import pandas as pd
from sklearn.utils import class_weight
import os
import numpy as np
import gc
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, Flatten, concatenate, Input
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


"""
    Author:
    Salman Ahmed
"""


SHAPE = 8000


def load_data(path):
    return pd.read_hdf(path)


def get_filenames(path):
    return os.listdir(path)


def get_data(SHAPE):
    paths = os.listdir(os.path.curdir)
    data = pd.DataFrame()
    for each in paths:
        n_path = os.path.join(os.path.curdir, each)
        data = pd.concat([data, pd.read_hdf(n_path)])
        if data.shape[0] >= SHAPE:
            return data
    return data


data = get_data(SHAPE)
data.pop('id')
data.pop('profile')
data.pop('driver_type')
data.pop('observe_net_shape')
data.pop('key')
data = data[data.action != 'AbortLaneChange']
ob_netCols = []
k = 0
while k < 465:
    ob_netCols.append("new_col"+str(k))
    k += 1
ob_net = data[ob_netCols]
for each in ob_netCols:
    data.pop(each)
y = data.pop('action')
np.unique(y)
ob_net = ob_net.reindex()
data = data.reindex()
ob_net1 = np.reshape(ob_net.values, (ob_net.shape[0], 155, 3))
data_features = getFeaturesTraining(data, ob_net1)
data = np.column_stack([data.values, data_features])
data = pd.DataFrame(data)
data = data.infer_objects()
data = data.reindex()
data[data.columns[0]] = data[data.columns[0]].astype('category')
data[data.columns[4]] = data[data.columns[4]].astype('category')
model_params1 = {
            'device': 'cpu', 
        "boosting_type": "gbdt", 
        "learning_rate": 0.09,
        "class_weight" : "balanced",
        "max_depth": -1,
        "num_leaves": 140,
        "n_estimators": 881,
        "bagging_fraction": 0.65,
        "feature_fraction": 0.48,
        "bagging_freq": 8,
        "bagging_seed": 2000,
        'min_child_samples': 40, 
        'min_child_weight': 100.0, 
        'min_split_gain': 0.1, 
        'reg_alpha': 0.008, 
        'reg_lambda': 0.08, 
        'subsample_for_bin': 18000, 
        'min_data_per_group': 100, 
        'max_cat_to_onehot': 4, 
        'cat_l2': 25.0, 
        'cat_smooth': 2.0, 
        'max_cat_threshold': 32, 
        "random_state": 12,
        "silent": True,
    }


def get_model_cnn():
    # Model1
    inp1 = Input(shape=(155, 3, 1))
    temp = Conv2D(64, (1, 1))(inp1)
    temp = Activation('relu')(temp)
    temp = Conv2D(32, (1, 1))(temp)
    temp = Activation('relu')(temp)
    temp = Conv2D(16, (1, 1))(temp)
    temp = Activation('relu')(temp)
    temp = Flatten()(temp)
    temp = Dense(32, name="get_features", activation='relu')(temp)
    
    # Model2
    inp2 = Input(shape=(47, 1))
    temp1 = Dense(32, activation='relu')(inp2)
    temp1 = Dense(16, activation='relu')(temp)
    # Merging
    merge = concatenate([temp, temp1])
    temp = Dense(12, activation='relu', name = "get_feature_map")(merge)
    temp = Dense(5)(temp)
    temp = Activation('sigmoid')(temp)
    model = Model([inp1, inp2], temp)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: %f — val_precision: %f — val_recall %f" %(_val_f1, _val_precision, _val_recall))
        return


metrics = Metrics()


def modeling_cross_validation(params, X, ob_net , y, nr_folds=5):
    clfs = list()
    nnclfs = list()
    ob_net = ob_net + 10.0
#    ob_net = ob_net/47.0
    kfolds = StratifiedKFold(n_splits=nr_folds, shuffle=True, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        ob_X_train, X_train, y_train = ob_net.iloc[trn_idx], X.iloc[trn_idx], y.iloc[trn_idx]
        ob_X_valid, X_valid, y_valid = ob_net.iloc[val_idx], X.iloc[val_idx], y.iloc[val_idx]
        print ("Training CNN for Fold "+str(n_fold+1))
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
        yFac = y_train
        yFac = to_categorical(yFac)
        ob_X_train = np.reshape(ob_X_train.values, (ob_X_train.shape[0], 155, 3, 1))
        ob_X_valid = np.reshape(ob_X_valid.values, (ob_X_valid.shape[0], 155, 3, 1))
        yFac1 = y_valid
        yFac1 = to_categorical(yFac1)
        model = get_model_cnn()
        trainNN = X_train.copy()
        validNN = X_valid.copy()
        trainNN = np.reshape(trainNN.values,(trainNN.shape[0],47,1))
        validNN = np.reshape(validNN.values,(validNN.shape[0],47,1))
        model.fit([ob_X_train, trainNN], yFac, class_weight=class_weights, epochs=1, batch_size=2048)
        nnclfs.append(model)
        layer_name = 'get_feature_map'
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        feature_map_train = intermediate_layer_model.predict([ob_X_train, trainNN])
        feature_map_valid = intermediate_layer_model.predict([ob_X_valid, validNN])
        del trainNN, validNN, ob_X_valid, ob_X_train
        gc.collect()
        X_train1 = X_train.copy()
        X_valid1 = X_valid.copy()
        X_train1 = np.column_stack([X_train1.values, feature_map_train])
        X_valid1 = np.column_stack([X_valid1.values, feature_map_valid])
        print("Fold {}".format(n_fold+1))
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train1, y_train,
            eval_set=[(X_valid1, y_valid)],
            verbose=200,
            early_stopping_rounds=150
        )
        clfs.append(model)
        fscore = f1_score(y_valid, model.predict(X_valid1), average='weighted')
        print("F1-Score ", fscore)
        accscore = accuracy_score(y_valid, model.predict(X_valid1))
        recscore = recall_score(y_valid, model.predict(X_valid1), average='weighted')
        prescore = precision_score(y_valid, model.predict(X_valid1), average='weighted')
        score = fscore
        confMat = confusion_matrix(y_valid, model.predict(X_valid1))
        print ("Accuracy ", accscore)
        print ("Recall ", recscore)
        print ("Precision ", prescore)
        print ("Confusion Matrix ")
        print (confMat)        
    return clfs, nnclfs


data = data.reindex()
ob_net = ob_net.reindex()
data = pd.get_dummies(data)
y = pd.factorize(y)[0]
y = pd.DataFrame(y)
clfs, nnclfs = modeling_cross_validation(model_params1, data, ob_net, y)
