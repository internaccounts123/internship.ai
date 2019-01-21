
import numpy as np
import pandas as pd
import csv
import json
import os
class Data_Saver:
    """
    Parallelised implementation of Data saving class
    """
    def __init__(self,filename,save_type,out):
        """
        :param filename:  Name of the output file
        :param save_type: h5,npy or csv
        :param out: Abs Path of the output Directory
        """
        self.current_rows=0
        self.file_name=filename
        self.save_type=save_type
        self.output_dir=out

    def save(self,Q):
        """
        Saves data in a specified format
        :param Q: Mutiprocessing Queue which serves a purpose of shared memory
        :return: Nothing
        """
        while True:
            print('Running_SAVE')
            self.filename=os.path.join(self.output_dir, str(self.file_name) + '.' + self.save_type)
            dataframe=Q.get(True,None)
            print ('Data_received')
            if (type(dataframe) is bool):
                print ('Broken')
                break
            if (self.filename[-4:]=='.csv'):
                self.save_csv(dataframe)
            elif(self.filename[-3:]=='.h5'):
                self.save_h5(dataframe)
            elif(self.filename[-4:]=='.npy'):
                self.save_npy(dataframe)
            self.file_name+=1
            print ('saved')
    def save_h5(self,dataframe):
        """
        Saves data to hard drive in h5 format
        :param dataframe: Pandas data frame to save
        :return: Nothing
        """
        dataframe['key']=pd.Series(np.arange(0,dataframe.shape[0],dtype=int))
        print( self.filename)
        dataframe.to_hdf(self.filename,key='key')
        print ('saved   ')
    def save_npy(self,dataframe):
        """
        Saves data in npy format and its columns in json format
        :param dataframe: Pandas data frame to save
        :return: Nothing
        """
        dataframe['key'] = pd.Series(np.arange(0, dataframe.shape[0], dtype=int))
        array=dataframe.values
        np.save(self.filename,array)
        columns=list(dataframe.columns)
        column_dict={i:columns[i] for i in range(len(columns))}
        with open('np_keys.json', 'w+') as fp:
            json.dump(column_dict, fp)


    def save_csv(self, data, mode='w'):
        """
        Saves csv to hard drive
        :param data: Pandas Data frame
        :param mode: a: append, w: write
        :return:Nothing
        """
        filename=self.filename
        dataframe['key'] = pd.Series(np.arange(0, dataframe.shape[0], dtype=int))
        if mode == 'a':  # Mode for appending Data
            data = data.values  # Dataframe to Numpy Array
            fields = data.tolist()  # Numpy Array to Python List
            with open(filename, 'a', newline='') as f:  # Opening given filename to append in it
                writer = csv.writer(f)  # Initillizing writer to write in it
                writer.writerows(fields)  # Write all data in rows and append in already created CSV
        elif mode == 'w':  # Mode for creating file and saving it as filename
            data.to_csv(filename, index=False)  # Saving dataframe without index
