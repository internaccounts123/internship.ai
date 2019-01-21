import os
import numpy as np
import keras
import glob
import pandas as pd
from multiprocessing import Queue
class Generator(keras.utils.Sequence):
    """
    Multiprocessing implementation of Keras Generator to read Data and feed to network
    """
    def __init__(self, config):
        """

        :param config:Network configurations
        """
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
        """
        Put files into Multiprocessing Queue
        :return:Nothing
        """
        files_batch_size = self.config['file_batch_size']
        files=self.DataFiles[self.Indexes_files[self.i_f:self.i_f+files_batch_size]]
        self.Data.put(self.load_files(files, self.config['format_']))
        self.i_f += files_batch_size

    def load_files(self, files, format_):
        """

        :param files:List of files to load
        :param format_:csv or h5
        :return:nd array of data loaded
        """
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
        return np.array(data_list)
    def generate(self):
        """
        Fills Data into multiprocessing queue when there's space availabe in a queue
        :return: Nothing
        """
        while self.Run==True:
            if self.Data.qsize()>=self.config['max_queue_size']:
                continue
            if self.i_f >= len(self.Indexes_files):
                np.random.shuffle(self.Indexes_files)
                self.i_f = 0
            self.fill_buffer()
    def load_data(self):
        """
        Fetch data from multiprocessing queue
        :return: nd array of data
        """
        self.examples_batch_size = self.config['ex_batch_size']
        if self.i_e >= len(self.Data_selected):
            self.Data_selected=self.Data.get(block=True,timeout=None)
            self.indexes_examples = np.arange(0,len(self.Data_selected),dtype= np.int32)
            np.random.shuffle(self.indexes_examples)
            self.i_e = 0
        res=self.Data_selected[self.indexes_examples[self.i_e:self.i_e+self.examples_batch_size]]
        self.i_e += self.examples_batch_size
        return res
    def __len__(self):
        """

        :return:  Total mini batches in an epoch
        """
        return (len(self.DataFiles)*self.config['file_examples'])//self.config['ex_batch_size']
    def on_epoch_end(self):
        pass
    def __getitem__(self, idx):
        """

        :param idx: idx of mini-batch(Not used but required by keras)
        :return: (X,y) to network
        """
        data = self.load_data()
        data1 = data[:,(~self.action_col)]
        action=data[:,self.action_col]
        return data1,keras.utils.to_categorical(action,self.actions_count)
