#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def reward(observation, action):
    reward = 0
    optimal_speed = observation[-3] - 2 # Speedlimit - 2
    speed = observation[-4]
    safe_net_front = observation[-5]
    car_start_index = 314   # our cars starting index in obnet
    car_front_index = car_start_index - 15 
    if speed < optimal_speed and action == 'Accelerate':
        reward = reward + 1
    elif speed > optimal_speed and action == 'Decelerate':
        reward = reward + 1
    # if we have invalid location on our car's front
    if  observation[car_front_index] == -10 :
        reward = reward - 500
        return reward
    
    safe = car_front_index
    for i in range(int(safe_net_front)):
        if observation[safe] != -1:
            reward=reward - 500
            return reward
        safe=safe-3
    
    # if we are on left lane and our action is leftlanechange
    if (observation[car_start_index-1] == observation[car_start_index-4] == observation[car_start_index-7] == -10) and action == 'LeftLaneChange':    
        reward = reward - 500
        return reward
    
    if (observation[car_start_index+1] == observation[car_start_index+4] == observation[car_start_index+7] == -10) and action == 'RightLaneChange':
        reward = reward - 500
        return reward
    # if our car's blocks have overlapping values collision has occured
    if observation[car_start_index] == observation[car_start_index - 3] == observation[car_start_index - 6] == observation[car_start_index - 9] == observation[car_start_index - 12]:
            reward += 0
    else:
        reward = reward - 500
        return reward
    return reward

