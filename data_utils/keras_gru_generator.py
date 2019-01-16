import os
import numpy as np
import keras
import glob
import pandas as pd
from multiprocessing import Queue
class Generator(keras.utils.Sequence):
    def __init__(self, config):
        self.config = config
        self.actions_count=self.config['labels']
        self.DataDirectory = self.config['data_directory']
        self.DataFiles = np.array(glob.glob(self.DataDirectory+'/*'+self.config['format_']))
        self.Indexes_files = np.array(list(range(len(self.DataFiles))))
        self.P=self.config['Preprocessor']
        self.action_col=config['action_col']
        self.Data = Queue()
        self.Data_selected=[]
        self.i_e = 0
        self.Run=True
        self.i_f = 0

    def fill_buffer(self):
        files_batch_size = self.config['file_batch_size']
        files=self.DataFiles[self.Indexes_files[self.i_f:self.i_f+files_batch_size]]
        self.Data.put(self.load_files(files, self.config['format_']))
        self.i_f += files_batch_size

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
            for i in files:
                if i[-3:] == '.h5':
                    array = pd.read_hdf(os.path.join(self.DataDirectory, i))
                    array=self.P.process_batch(array)
                    self.obsnet_col=array.columns=='ob_net'
                    self.action_col=array.columns=='action'
                    data_list.extend(array.values)
        elif format_ == 'npy':
            for i in files:
                if i[-4:] == '.csv':
                    array = np.load(os.path.join(self.DataDirectory, i))
                    data_list.extend(array)
        return np.array(data_list)
    def generate(self):
        while self.Run==True:
            if self.Data.qsize()>=self.config['max_queue_size']:
                continue
            if self.i_f >= len(self.Indexes_files):
                np.random.shuffle(self.Indexes_files)
                self.i_f = 0
            self.fill_buffer()
    def load_data(self):
        self.examples_batch_size = self.config['ex_batch_size']
        if self.i_e >= len(self.Data_selected):
            self.Data_selected=self.Data.get(block=True,timeout=None)
            self.indexes_examples = np.arange(0,len(self.Data_selected),dtype= np.int32)
            np.random.shuffle(self.indexes_examples)
            self.i_e = 0
        res=self.Data_selected[self.indexes_examples[self.i_e:self.i_e+self.examples_batch_size*self.config['states']]]
        if (res.shape[0]<self.examples_batch_size*self.config['states']):
            z=np.zeros((res.shape[0]-self.examples_batch_size*self.config['states'],res.shape[1]))
            res=np.concatenate(res,z)
        self.i_e += self.examples_batch_size*self.config['states']
        return res.reshape(self.config['states'],self.examples_batch_size,-1)
    def __len__(self):
        return (len(self.DataFiles)*self.config['file_examples'])//self.config['ex_batch_size']
    def on_epoch_end(self):
        pass
    def __getitem__(self, idx):
        data = self.load_data()
        data1 = data[...,(~self.action_col)]
        action=data[...,self.action_col].flatten()
        return data1,keras.utils.to_categorical(action,self.actions_count).reshape(data1.shape[0],data1.shape[0],-1)
