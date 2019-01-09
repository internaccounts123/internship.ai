#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn import preprocessing
import pickle

class preprocessing:
    
    def __init__(self,dir_path):
        self.batch_count=0
        self.dir_path=dir_path
    
    def process_batch(batch):
        batch=self.remove_features(batch)
        batch=flatten_obnet(batch)
        return batch
    
    def remove_features(batch):
            batch=batch.drop(['observe_net_shape','profile'],axis=1)
            if 'key' in batch.columns:
                batch=batch.drop(['key'],axis=1)
        return batch
    def flatten_obnet(batch):
        batch['ob_net']=batch['ob_net'].values.flatten()
        return batch
    
    def categorical_features(batch)
        self.features=[]
        for feature in batch.columns: 
            if batch[feature].dtype=='object':
                self.features.append(feature)
        
    def save_label_encoder(batch):
        
        features=categorical_features(batch)
        encoder_index=1
        for feature in features:
            le = preprocessing.LabelEncoder()
            le.fit(batch[feature])
            pickle.dump(le,dir_path+"/labelencoder"+encoder_index+".pkl")
            encoder_index+=1
        
    def load_label_encoder():
        encoder_index=1
        label_encoders=[]
        for i in range(len(self.features)):
            with open(dir_path+"/labelencoder"+encoder_index+".pkl", 'rb') as f:      
                label_encoders.append(pickle.load(f)) 
        return label_encoders
    
    def process_batch(batch):
        self.batch_count+=1
         

