import os 

def main(save_type=='csv'):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    #get paths of all the pkl files 
    paths=get_pkl(directory = dir_path, max_depth=-1)
    #make generator that contains all the batches 
    generator=data_generator(paths,1)
    for batch in batches:
        #convert each batch 
        dataframe=conv_to_pd_dataframe(batch)
        #save each batch
        save(dataframe,save_type)
    print (dir_path)
  
if __name__=="__main__":
    main()
