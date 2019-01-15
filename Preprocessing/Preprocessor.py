from sklearn import preprocessing
import pickle
import pandas as pd

class preprocessor:
    
    """
    The preprocessor class that performs preprocessing on data such as removing unwanted features,
    saving and loading label encoder, label encoding categorical data, normalizing data and getting train test
    split.
    
    """
    
    def __init__(self,dir_path):
        
        #Batch count is used to calculate train test split
        self.batch_count=0
        self.dir_path=dir_path
    
    def process_batch(self,batch):
        
        self.batch_count+=1
        batch=self.remove_features(batch)
        batch=batch.reindex(columns=sorted(batch.columns))
        batch[batch['action']=='Unknown'].action='Accelerate'
        batch[batch['previous_decision']=='Unknown'].previous_decision='Accelerate'
        label_encoders=self.load_label_encoder()
        batch=self.label_encode(label_encoders,batch)
        return batch
    
    
    
    def remove_features(self,batch):
        
        #Remove unwanted features
        batch=batch.drop(['observe_net_shape','profile','driver_type'],axis=1)
        if 'key' in batch.columns:
            batch=batch.drop(['key'],axis=1)
        return batch
    
    
    def categorical_features(self,batch):
        
        #Get categorical features for label encoding
        self.features=[]
        for feature in batch.columns: 
            if batch[feature].dtype=='object' and feature!="ob_net":
                self.features.append(feature)
        
    def save_label_encoder(self,batch):
        
        #Save label encoders for each categorical feature
        self.categorical_features(batch)
        for feature in self.features:
            le=preprocessing.LabelEncoder()
            le.fit(batch[feature])
            pickle.dump(le,open(self.dir_path+"/"+feature+".pkl",'wb'))
        
    def load_label_encoder(self):
        
        #Load and return all label encoders
        label_encoders=[]
        for feature in ['action','lane_change_mode','previous_decision']:
            with open(self.dir_path+"/"+feature+".pkl", 'rb') as f:      
                label_encoders.append(pickle.load(f)) 
        return label_encoders
            
    def label_encode(self,label_encoders,batch):
        
        #Use the label encoders on new batches
        label_encoder_index=0
        columns=batch.columns
        for column in columns:
            if column!='ob_net':
                if batch[column].dtype=='object':
                    batch[column]=label_encoders[label_encoder_index].transform(batch[column])
                    label_encoder_index+=1
                
        for column in batch.columns:
            batch[column]=batch[column].astype(float)
        return batch
    def train_test_split(self):
        #Return train test split batches with a ratio of 75:25 for train and test
        return int(self.batch_count*0.75),int(self.batch_count*0.25)

