import os
import sys
import pickle
import numpy as np
import pandas as pd

def create_batch(file_paths,batch_size,file_index,starting_index):
    batch=[]
    loop=True
    while loop:
        with open(file_paths[file_index], 'rb') as f:
            file=(list(pickle.load(f))) 
            if len(file[starting_index:])+len(batch)<batch_size:
                batch.extend(file[starting_index:])
                file_index+=1
                starting_index=0
            else:
                appending_examples=batch_size-len(batch)
                batch.extend(file[starting_index:starting_index+appending_examples])
                starting_index+=appending_examples
                loop=False
                
    return batch,file_index,starting_index        

def data_generator(file_paths,batch_size):
    #file_paths contains the absolute paths of all the files we need to convert and save
    #batch_size tells us how many files to load and save in a single go(depends upon the size of RAM)
    
    file_index=0
    starting_index=0
    
    while True:
        batch,file_index,starting_index=create_batch(file_paths,batch_size,file_index,starting_index)   
        yield batch
        if file_index>=len(file_paths):
            break 
        


def main(directory= os.path.dirname(os.path.realpath(__file__)),save_type='csv',batch_size=100000):

    #convert our arguments into integers 
    batch_size=int(batch_size)
    #get file path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #get paths of all the pkl files 
    paths=get_pkl(directory = dir_path, max_depth=-1)
    #make generator that contains all the batches 
    batches=data_generator(paths,batch_size)
    file_name=1
    for batch in batches:
            #convert each batch 
            dataframe=conv_to_pd_dataframe(batch)
            filename='output.h5'
            D=Data_Saver(os.path.join(directory,str(file_name)+'.'+save_type))
            D.save(dataframe)
            file_name+=1
          
if __name__=="__main__":
    if len(sys.argv)==2:
        main(sys.argv[1])
    elif len(sys.argv)==3:
        main(sys.argv[1],sys.argv[2])
    elif len(sys.argv)==4:
        main(sys.argv[1],sys.argv[2],sys.argv[3])
    else:
        main()


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
        dataframe['key']=pd.Series(np.arange(0,temp_pd.shape[0],dtype=int))
        dataframe.to_hdf(self.filename,key='key')
    def save_npy(self,dataframe):
        array=dataframe.values
        np.save(self.filename,array)      
D=Data_Saver(filename='output.h5')
D.save(temp_pd)

