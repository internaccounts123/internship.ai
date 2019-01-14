import numpy as np

import pandas as pd

from collections import defaultdict

def conv_to_pd_dataframe(Input_Batch_Queue, Output_Batch_Queue,col_names):
    observations_list=True
    while True:
        print ('Running')
        observations_list=Input_Batch_Queue.get(block=True,timeout=None)
        if (observations_list==False):
            Output_Batch_Queue.put(False)
            break
        if (type(observations_list[0])==dict):
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
            final_df=pd.DataFrame(result_set)
        elif (type(observations_list[0])==list):
            cols = ['id', 'lane_change_mode']
            k = 0
            while k < 465:
                cols.append("new_col"+str(k))
                k+=1
            cols.extend(['safe_net_front', 'action',
                   'speed', 'speed_limit', 'profile', 'driver_type',
                   'previous_decision', 'previous_speed', 'observe_net_shape'])
            print (len(observations_list[0]))
            final_df=pd.DataFrame(data=observations_list,columns=cols)
        print ('Pushing data')
        Output_Batch_Queue.put(final_df)


#python main.py C:\Users\DELL\Documents\AAI\new_data_h5 h5 10000 C:\Users\DELL\Documents\AAI\new_data