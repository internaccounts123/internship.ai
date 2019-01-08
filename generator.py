import pickle

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
        