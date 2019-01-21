import joblib
import pandas as pd
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
import gc
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb


"""
    Author:
    Salman Ahmed
"""


def get_data(no_file=1):
    paths = os.listdir(os.path.curdir)
    data = pd.DataFrame()
    for each in paths:
        print (data.shape)
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
        temp = temp[temp.driver_type=='Rule']
        temp = temp[temp.action!='AbortLaneChange']
        if temp.shape[0]>0:
            temp.pop('driver_type')
            data = pd.concat([data,temp])
    return data


data = get_data(1000000)

data['previous_decision'] = data['previous_decision'].astype('category')
data['action'] = data['action'].astype('category')
data['lane_change_mode'] = data['lane_change_mode'].astype('category')

model_params1 = {
            'device': 'cpu', 
        "boosting_type": "gbdt", 
        "learning_rate": 0.085,
        "class_weight" : "balanced",
        "max_depth": 15,
        "num_leaves": 140,
        "n_estimators": 121,
        "bagging_fraction": 0.75,
        "feature_fraction": 0.61,
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


def modeling_cross_validation(params, X, y, nr_folds=5):
    clfs = list()
    tempk = 0
    kfolds = StratifiedKFold(n_splits=nr_folds, shuffle=True, random_state=42)
    for n_fold, (trn_idx, val_idx) in enumerate(kfolds.split(X, y)):
        X_train, y_train = X.iloc[trn_idx], y.iloc[trn_idx]
        X_valid, y_valid = X.iloc[val_idx], y.iloc[val_idx]

        print("Fold {}".format(n_fold+1))
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=200,
            early_stopping_rounds=150
        )
        clfs.append(model)
        fscore = f1_score(y_valid, model.predict(X_valid), average='weighted')
        print("F1-Score ", fscore)
        accscore = accuracy_score(y_valid, model.predict(X_valid))
        recscore = recall_score(y_valid, model.predict(X_valid), average='weighted')
        prescore = precision_score(y_valid, model.predict(X_valid), average='weighted')
        score = fscore
        confMat = confusion_matrix(y_valid, model.predict(X_valid))
        print("Accuracy ", accscore)
        print("Recall ", recscore)
        print("Precision ", prescore)
        print("Confusion Matrix ")
        print(confMat)
        tempk += 1
    return clfs


y = data.pop('action')
y = pd.factorize(y)[0]
y = pd.DataFrame(y)
clfs = modeling_cross_validation(model_params1, data, y)
k = 0
for each in clfs:
    joblib.dump(each, "LGB_"+str(k))
    k += 1

