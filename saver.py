
import numpy as np
import pandas as pd
import csv
import json

class Data_Saver:
    def __init__(self,filename):
        self.current_rows=0
        self.filename=filename
    def save(self,dataframe):
        if (self.filename[-4:]=='.csv'):
            self.save_csv(dataframe)
        elif(self.filename[-3:]=='.h5'):
            self.save_h5(dataframe)
        elif(self.filename[-4:]=='.npy'):
            self.save_npy(dataframe)
    def save_h5(self,dataframe):
        dataframe['key']=pd.Series(np.arange(0,dataframe.shape[0],dtype=int))
        dataframe.to_hdf(self.filename,key='key')
    def save_npy(self,dataframe):
        array=dataframe.values
        np.save(self.filename,array)
		columns=list(dataframe.columns)
        column_dict={i:columns[i] for i in range(len(columns))}
        with open('np_keys.json', 'w+') as fp:
            json.dump(column_dict, fp)


    def save_csv(self, data, mode='w'):
        """
        @Author: Salman Ahmed

            data is a dataframe having rows in it to be appended or saved.
            filename is name of file where data should be saved.
            mode, if mode is w that means we are writing it first time
            if mode is a then it will append in already created file.
        """
        filename=self.filename
        if mode == 'a':  # Mode for appending Data
            data = data.values  # Dataframe to Numpy Array
            fields = data.tolist()  # Numpy Array to Python List
            with open(filename, 'a', newline='') as f:  # Opening given filename to append in it
                writer = csv.writer(f)  # Initillizing writer to write in it
                writer.writerows(fields)  # Write all data in rows and append in already created CSV
        elif mode == 'w':  # Mode for creating file and saving it as filename
            data.to_csv(filename, index=False)  # Saving dataframe without index
