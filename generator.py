import pickle

def create_batch(file_paths,batch_size,file_index,starting_index):
    
    """
    Creates a single batch and passes it to data generator function
    
    args:
    file_paths:     The absolute paths of all the input files
    batch_size:     The number of examples to process in one batch
    file_index:     Current input file that our batch will be reading 
    starting_index: example of the input file our batch starts reading from 
    
    returns:
    A single batch, file_index, starting_index
    """
    
    batch=[]         
    loop=True
    while loop and len(file_paths)>file_index:             #While our batch is being filled and while we still have files to read
        with open(file_paths[file_index], 'rb') as f:      #Open a new file
            file=(list(pickle.load(f))) 
            if len(file[starting_index:])+len(batch)<batch_size: #if file is completely read and batch is not full
                batch.extend(file[starting_index:])
                file_index+=1
                starting_index=0
            else:                                                #if batch does not have capacity to fill the whole file
                batch_capacity=batch_size-len(batch)         #the capacity of unfilled batch
                batch.extend(file[starting_index:starting_index+batch_capacity])
                starting_index+=batch_capacity
                loop=False
                
    return batch,file_index,starting_index        

def data_generator(file_paths,batch_size):
    
    """
    Python generator that generates one batch at a time, the size of which depends upon the batch size specified
    
    args:
    file_paths: The absolute paths of all the input files
    batch_size: The number of examples to process in one batch
    
    returns:
    Batches with examples one at a time
    """
    
    file_index=0                         #current input file that our batch will be reading 
    starting_index=0                     #example of the input file our batch starts reading from 
    
    while True:
        batch,file_index,starting_index=create_batch(file_paths,batch_size,file_index,starting_index)   
        yield batch
        if file_index>=len(file_paths):  #all the files have been read and we will thus exit the generator
            break 
