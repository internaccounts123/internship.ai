import pickle

def create_batch(file_paths,batch_size,batch_index):
    batch=[]
    for i in range(batch_size):
            scaling=batch_index*batch_size+i
            with open(file_paths[scaling], 'rb') as f:
                batch.append(list(pickle.load(f)))
    return batch        

def data_generator(file_paths,batch_size=1,fast=False):
    #file_paths contains the absolute paths of all the files we need to convert and save
    #batch_size tells us how many files to load and save in a single go(depends upon the size of RAM)
    
    num_batches=len(file_paths)/batch_size
    assert (num_batches==int(num_batches))
    num_batches=int(num_batches)
    
    for i in range(num_batches):
        batch=create_batch(file_paths,batch_size,i)   
        yield batch



