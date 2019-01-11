import numpy as np
import pandas as pd
from collections import defaultdict
def conv_to_pd_dataframe(Input_Batch_Queue, Output_Batch_Queue):
    #inuts: 
    #observation_list: list of file observations
    #output
    #pd dataframe  that has all rows appnded
    observations_list=True
    while True:
        print ('Running')
        observations_list=Input_Batch_Queue.get(block=True,timeout=None)
        if (observations_list==False):
            Output_Batch_Queue.put(False)
            break
        keys=list(observations_list[0].keys())
        keys_set=set(keys)
        result_set=defaultdict(list)
        num_observations=len(observations_list)
        for j in range(num_observations):
            observation=observations_list[j]
            observation_keys=list(observation.keys())
            assert keys_set==set(observation_keys)
            for key in keys:
                if key=="ob_net":
                    result_set[key].append(observation[key].tolist())
                else:
                    result_set[key].append(observation[key])
        print ('Pushing data')
        Output_Batch_Queue.put(pd.DataFrame(result_set))