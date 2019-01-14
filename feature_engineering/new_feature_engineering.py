def speedGap(speedLimit, curSpeed):
    """
        Returns:
            Returns Gap in Speed Limit and Current Speed
        Parameters:
            speedLimit -> Max Limit of Speed
            curSpeed -> Current Speed of Agent
    """

    return (speedLimit-curSpeed)

def maxValue(row):
    """
        Returns:
            Returns Maximum Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """

    return np.max(row, axis=1)

def minValue(row):
    """
        Returns:
            Returns Minimum Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """
    return np.min(row, axis=1)

def avgValue(row):
    """
        Returns:
            Returns Average Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """
    return np.average(row, axis=1)

def sumValue(row):
    """
        Returns:
            Returns Sum Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """
    return np.sum(row, axis=1)

def find_features_using_obnet(ob_net,num_blocks_1_car, num_lanes=3):    
    ob_net=(np.stack(data.ob_net.tolist()))
    assert len(ob_net.shape)==3    
    ob_net=np.transpose(ob_net,axes=(0,2,1))
    front_rows=ob_net[:,:,:100]
    rear_rows=ob_net[:,:,105:155]
    front_mask=front_rows>=0
    rear_mask=rear_rows>=0    #num cars in front (all lanes)
    sum_front_mask=np.sum(front_mask,axis=2)
    num_blocks_incomp_car=sum_front_mask%num_blocks_1_car
    comp_cars=(sum_front_mask/num_blocks_1_car).astype(np.int)
    incomp_car_present=num_blocks_incomp_car>0    
    num_cars_front=comp_cars+(incomp_car_present)    #num cars in rear (all lanes)
    sum_rear_mask=np.sum(rear_mask,axis=2)
    num_blocks_incomp_car=sum_rear_mask%num_blocks_1_car
    comp_cars=(sum_rear_mask/num_blocks_1_car).astype(np.int)
    incomp_car_present=num_blocks_incomp_car>0    
    num_cars_rear=comp_cars+(incomp_car_present)    #avg speeds in front
    sum_front_valid_speeds=np.sum(front_rows*front_mask,axis=2)
    avg_speed=sum_front_valid_speeds+((front_rows[:,:,0]*(num_blocks_1_car-num_blocks_incomp_car))*incomp_car_present)
    avg_speed=avg_speed/(num_cars_front*num_blocks_1_car)
    avg_speed[num_cars_front==0]=0    #distance to the closest car in front in each lane
    front_distances_all_lanes=np.arange(100,0,-1)*front_mask
    front_distances_all_lanes[front_distances_all_lanes==0]=100
    front_distances_all_lanes=np.min(front_distances_all_lanes,axis=2)    #distance to the closest car in front in each lane
    rear_distances_all_lanes=np.arange(1,51)*rear_mask
    rear_distances_all_lanes[rear_distances_all_lanes==0]=50
    rear_distances_all_lanes=np.min(rear_distances_all_lanes,axis=2)
    lanes_closed_in_front=(front_rows==-10).any(axis=2)    #another car is in our safe_net front
    in_safe_net_front=np.logical_and( (front_distances_all_lanes<=np.expand_dims(data.safe_net_front,axis=-1)), (rear_distances_all_lanes<=np.expand_dims(data.safe_net_front/2.0,axis=-1)))    #speed of closest car in front
    num_examples=ob_net.shape[0]
    mask_cars_not_present=num_cars_front==0
    e_i=np.tile(np.expand_dims(np.arange(num_examples),axis=1),reps=[1,num_lanes])
    l_i=np.tile(np.arange(num_lanes),reps=[num_examples,1])
    front_closest_speed=(front_rows[:,:,::-1])[e_i,l_i,front_distances_all_lanes-1]
    front_closest_speed[mask_cars_not_present] =-1    #speed of closest car in rear
    mask_cars_not_present=num_cars_rear==0
    e_i=np.tile(np.expand_dims(np.arange(num_examples),axis=1),reps=[1,num_lanes])
    l_i=np.tile(np.arange(num_lanes),reps=[num_examples,1])
    rear_closest_speed=(rear_rows[:,:,:])[e_i,l_i,rear_distances_all_lanes-1]
    rear_closest_speed[mask_cars_not_present] =-1    #speed of closest car in rear
    pressure_front = num_cars_front.copy()
    pressure_front[:,0] = pressure_front[:,0]/20.0
    pressure_front[:,1] = pressure_front[:,1]/20.0
    pressure_front[:,2] = pressure_front[:,2]/20.0
    pressure_rear = num_cars_rear.copy()
    pressure_rear[:,0] = pressure_rear[:,0]/5.0
    pressure_rear[:,1] = pressure_rear[:,1]/5.0
    pressure_rear[:,2] = pressure_rear[:,2]/5.0
    front_closest_speed_distance = front_closest_speed.copy()
    front_closest_speed_distance[:,0] = front_closest_speed_distance[:,0]*front_distances_all_lanes[:,0]
    front_closest_speed_distance[:,1] = front_closest_speed_distance[:,1]*front_distances_all_lanes[:,1]
    front_closest_speed_distance[:,2] = front_closest_speed_distance[:,2]*front_distances_all_lanes[:,2]
    return pressure_front,num_cars_front,num_cars_rear, avg_speed,front_distances_all_lanes, rear_distances_all_lanes,lanes_closed_in_front,in_safe_net_front,front_closest_speed,rear_closest_speed, front_closest_speed_distance



def getFeaturesTraining(data):
    data.ob_net=data.ob_net.apply(lambda x: np.array(x))
    getFeatures = find_features_using_obnet(data.ob_net, 5)
    pressure_front,num_cars_front,num_cars_rear, avg_speed,front_distances_all_lanes, rear_distances_all_lanes,lanes_closed_in_front,in_safe_net_front,front_closest_speed,rear_closest_speed, front_closest_speed_distance = getFeatures
    speedgap = speedGap(data.speed_limit, data.speed)
    numericalColumns = ['speed_limit', 'speed', 'safe_net_front', 'previous_speed']
    maxvalue = maxValue(data[numericalColumns].values)
    minvalue = minValue(data[numericalColumns].values)
    avgvalue = avgValue(data[numericalColumns].values)
    sumvalue = sumValue(data[numericalColumns].values)
    return np.column_stack([front_closest_speed_distance, pressure_front,num_cars_front,num_cars_rear, avg_speed,front_distances_all_lanes, rear_distances_all_lanes,lanes_closed_in_front,in_safe_net_front,front_closest_speed,rear_closest_speed, speedgap, maxvalue, minvalue, avgvalue, sumvalue])
    
    
    