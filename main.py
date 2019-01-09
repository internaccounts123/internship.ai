import os
import sys
from get_paths_recursive import get_pkl
from generator import data_generator
from convertor import conv_to_pd_dataframe
from saver import Data_Saver
from multiprocessing import Process, Queue
def main(output_dir= os.path.dirname(os.path.realpath(__file__)),save_type='h5',batch_size=10000,dir_path=os.path.dirname(os.path.realpath(__file__))):
    
    """"
    The main function that takes arguments from the user, and uses the different functions to load and save data.
    
    args:
    output_dir:  The directory of the saved files
    save_type:   The data type of the saved files(csv,h5,npy)
    batch_size:  Number of examples in a batch
    dir_path:    The directory of input files
    
    """
    Input_Batch_Queue = Queue()
    Output_Batch_Queue = Queue()
    batch_size=int(batch_size)
    paths=get_pkl(directory = dir_path, max_depth=-1)
    #make generator that contains all the batches 
    batches=data_generator(paths,batch_size)
    p = Process(target=conv_to_pd_dataframe, args=(Input_Batch_Queue, Output_Batch_Queue))
    file_name=1
    for batch in batches:
            #convert each batch
            Input_Batch_Queue.put(batch)
            dataframe=conv_to_pd_dataframe(batch)
            #Data saver class
            D=Data_Saver(os.path.join(output_dir,str(file_name)+'.'+save_type))
            #Save data
            D.save(Output_Batch_Queue)
            #change file name
            file_name+=1
    Input_Batch_Queue.put(False)

if __name__=="__main__":
    
    #all these if's so that each argument can have a default value in main() 
    if len(sys.argv)==2:
        main(sys.argv[1])
    elif len(sys.argv)==3:
        main(sys.argv[1],sys.argv[2])
    elif len(sys.argv)==4:
        main(sys.argv[1],sys.argv[2],sys.argv[3])
    elif len(sys.argv)==5:
        main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
    else:
        main()
