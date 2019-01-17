import os
import numpy as np
import keras
import glob
import pandas as pd
from multiprocessing import Queue
from sklearn.utils import resample

class Generator(keras.utils.Sequence):
    def __init__(self, config):
        self.config = config
        self.actions_count=self.config['labels']
        self.DataDirectory = self.config['data_directory']
        self.DataFiles = np.array(glob.glob(self.DataDirectory+'/*'+self.config['format_']))
        self.Indexes_files = np.array(list(range(len(self.DataFiles))))
        self.P=self.config['Preprocessor']
        self.action_col=config['action_col']
        self.Data_selected=[]
        self.i_e = 0
        self.i_f = 0

    def fill_buffer(self):
        files_batch_size = self.config['file_batch_size']
        files=self.DataFiles[self.Indexes_files[self.i_f:self.i_f+files_batch_size]]
        self.Data_selected = self.load_files(files, self.config['format_'])
        self.mean = np.mean(self.Data_selected[:, (~self.action_col)], axis=0)
        self.std = np.std(self.Data_selected[:, (~self.action_col)], axis=0)+1e-10
        self.i_f += files_batch_size
        self.actions = self.Data_selected[:, self.action_col].flatten()
        (unique_actions, counts) = np.unique(self.actions, return_counts=True)
        splitted_data = []
        splitted_data_shapes=counts
        for action in unique_actions:
            temp=self.Data_selected[self.actions == action]
            splitted_data.append(temp)
        optimal_shape=splitted_data_shapes[np.argsort(splitted_data_shapes)[2]]
        new_data=[]
        for i in range(len(splitted_data)):
            new_data.extend(resample(splitted_data[i], n_samples=optimal_shape))
        self.Data_selected=np.array(new_data)
        self.indexes_examples = np.arange(0, len(self.Data_selected), dtype=np.int32)
        np.random.shuffle(self.indexes_examples)

    def load_files(self, files, format_):
        data_list = []
        if format_ == 'csv':
            for i in files:
                if i[-4:] == '.csv':
                    array = pd.read_csv(os.path.join(self.DataDirectory, i))
                    array=self.P.process_batch(array)
                    self.action_col=array.columns=='action'
                    data_list.extend(array.values)
        elif format_ == 'h5':
            for count,i in enumerate(files):
                if i[-3:] == '.h5':
                    print ('File No:',count)
                    array = pd.read_hdf(os.path.join(self.DataDirectory, i))
                    array=self.P.process_batch(array)
                    self.action_col=array.columns=='action'
                    data_list.extend(array.values.astype(np.float32))
        elif format_ == 'npy':
            for i in files:
                if i[-4:] == '.csv':
                    array = np.load(os.path.join(self.DataDirectory, i))
                    data_list.extend(array)
        return np.array(data_list)

    def generate(self):
        self.fill_buffer()

    def load_data(self):
        self.examples_batch_size = self.config['ex_batch_size']
        if self.i_e >= len(self.Data_selected):
            self.indexes_examples = np.arange(0,len(self.Data_selected),dtype= np.int32)
            np.random.shuffle(self.indexes_examples)
            self.i_e = 0
        res=self.Data_selected[self.indexes_examples[self.i_e:self.i_e+self.examples_batch_size]]
        self.i_e += self.examples_batch_size
        return res

    def __len__(self):
        return (self.config['file_batch_size']*self.config['file_examples'])//self.config['ex_batch_size']

    def on_epoch_end(self):
        pass

    def normalize(self,data):
        return (data-self.mean)/self.std
    def __getitem__(self, idx):
        data = self.load_data()
        data1 = data[:,(~self.action_col)]
        action=data[:,self.action_col]
        return self.normalize(data1),keras.utils.to_categorical(action,self.actions_count)