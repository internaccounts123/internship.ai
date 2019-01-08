import os
import sys

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
            save(dataframe,str(file_name)+'.'+save_type,directory)
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
