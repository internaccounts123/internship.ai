def speed_gap(speed_limit, cur_speed):
    """
        Returns:
            Returns Gap in Speed Limit and Current Speed
        Parameters:
            speedLimit -> Max Limit of Speed
            curSpeed -> Current Speed of Agent
    """

    return speed_limit - cur_speed


def max_value(row):
    """
        Returns:
            Returns Maximum Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """

    return np.max(row, axis=1)


def min_value(row):
    """
        Returns:
            Returns Minimum Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """
    return np.min(row, axis=1)


def avg_value(row):
    """
        Returns:
            Returns Average Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """
    return np.average(row, axis=1)


def sum_value(row):
    """
        Returns:
            Returns Sum Value across each example with Numerical Values
        Parameters:
            row -> Example with Numberical Attributes
    """
    return np.sum(row, axis=1)


def find_features_using_obnet(data, num_blocks_1_car, num_lanes=3):
    """
    @authors: Armughan and Salman 
        Finds additional features using obnet

        Vectorized code to improve speed

        Params:
            data: pandas dataframe (must include ob net columns and safe net front)
            num_blocks_1_car: blocks occupied by 1 car
            num_lanes: number of lanes  in the obnet
        Returns:
            All values have shape num_examples, num_lanes

            pressure_front : traffic intensity in each of the lanes (num_cars in front/max number of cars in each lane)
            num_cars_front : num cars in front of the agent  in each lane 
            num_cars_rear : num cars in behind the agent  in each lane 
            avg_speed : avg speed of cars in front of the agent in each lane
            front_distances_all_lanes :  y distance bw closest car (ahead of the agent ) and the agent in each lane (if no lane present 1000 is output)
            rear_distances_all_lanes : y distance bw closest car (behind the agent ) and the agent in each lane (if no lane present 1000 is output)
            lanes_closed_in_front : which of the lanes are closed in front
            in_safe_net_front : does the agent have a car in its safe net front
            front_closest_speed : speed of the closest car ahead of the agent in each of the lane
            rear_closest_speed : speed of the closest car behind the agent in each of the lane
            front_closest_speed_distance: product of speed and distance 

    """
    ob_net = (np.stack(data.ob_net.tolist()))
    assert len(ob_net.shape) == 3
    ob_net = np.transpose(ob_net, axes=(0, 2, 1))
    front_rows = ob_net[:, :, :100]
    rear_rows = ob_net[:, :, 105:155]
    front_mask = front_rows >= 0
    rear_mask = rear_rows >= 0    # num cars in front (all lanes)
    sum_front_mask = np.sum(front_mask, axis=2)
    num_blocks_incomp_car = sum_front_mask % num_blocks_1_car
    comp_cars = (sum_front_mask/num_blocks_1_car).astype(np.int)
    incomp_car_present = num_blocks_incomp_car > 0
    num_cars_front = comp_cars + incomp_car_present    # num cars in rear (all lanes)
    sum_rear_mask = np.sum(rear_mask, axis=2)
    num_blocks_incomp_car = sum_rear_mask % num_blocks_1_car
    comp_cars = (sum_rear_mask/num_blocks_1_car).astype(np.int)
    incomp_car_present = num_blocks_incomp_car > 0
    num_cars_rear = comp_cars + incomp_car_present    # avg speeds in front
    sum_front_valid_speeds = np.sum(front_rows*front_mask, axis=2)
    avg_speed=sum_front_valid_speeds+((front_rows[:, :, 0]*(num_blocks_1_car-num_blocks_incomp_car))*incomp_car_present)
    avg_speed=avg_speed/(num_cars_front*num_blocks_1_car)
    avg_speed[num_cars_front == 0] = 0    # distance to the closest car in front in each lane
    front_distances_all_lanes = np.arange(100, 0, -1)*front_mask
    front_distances_all_lanes[front_distances_all_lanes == 0] = 100
    front_distances_all_lanes = np.min(front_distances_all_lanes, axis=2)
    # distance to the closest car in front in each lane
    rear_distances_all_lanes = np.arange(1, 51)*rear_mask
    rear_distances_all_lanes[rear_distances_all_lanes == 0] = 50
    rear_distances_all_lanes = np.min(rear_distances_all_lanes, axis=2)
    lanes_closed_in_front = (front_rows == -10).any(axis=2)    # another car is in our safe_net front
    in_safe_net_front = np.logical_and((front_distances_all_lanes <= np.expand_dims(data.safe_net_front, axis=-1)),
                                       (rear_distances_all_lanes <= np.expand_dims(data.safe_net_front/2.0, axis=-1)))
    # speed of closest car in front
    num_examples = ob_net.shape[0]
    mask_cars_not_present=num_cars_front == 0
    e_i = np.tile(np.expand_dims(np.arange(num_examples), axis=1), reps=[1, num_lanes])
    l_i = np.tile(np.arange(num_lanes), reps=[num_examples, 1])
    front_closest_speed = (front_rows[:, :, ::-1])[e_i, l_i, front_distances_all_lanes-1]
    front_closest_speed[mask_cars_not_present] = -1    # speed of closest car in rear
    mask_cars_not_present = num_cars_rear == 0
    e_i = np.tile(np.expand_dims(np.arange(num_examples), axis=1), reps=[1, num_lanes])
    l_i = np.tile(np.arange(num_lanes), reps=[num_examples, 1])
    rear_closest_speed = (rear_rows[:, :, :])[e_i, l_i, rear_distances_all_lanes-1]
    rear_closest_speed[mask_cars_not_present] = -1    # speed of closest car in rear
    pressure_front = num_cars_front.copy()
    pressure_front=pressure_front/20.0
    # pressure_rear = num_cars_rear.copy()
    # pressure_rear=pressure_rear/5.0
    front_closest_speed_distance = front_closest_speed.copy()
    front_closest_speed_distance=front_closest_speed_distance*front_distances_all_lanes

    return pressure_front,num_cars_front,num_cars_rear, avg_speed,front_distances_all_lanes, rear_distances_all_lanes,\
           lanes_closed_in_front,in_safe_net_front,front_closest_speed,rear_closest_speed, front_closest_speed_distance


def get_features_training(data):
    data.ob_net = data.ob_net.apply(lambda x: np.array(x))
    get_features = find_features_using_obnet(data.ob_net, 5)
    pressure_front, num_cars_front, num_cars_rear, avg_speed, front_distances_all_lanes, rear_distances_all_lanes, lanes_closed_in_front, in_safe_net_front, front_closest_speed, rear_closest_speed, front_closest_speed_distance = get_features
    speedgap = speed_gap(data.speed_limit, data.speed)
    numerical_columns = ['speed_limit', 'speed', 'safe_net_front', 'previous_speed']
    maxvalue = max_value(data[numerical_columns].values)
    minvalue = min_value(data[numerical_columns].values)
    avgvalue = avg_value(data[numerical_columns].values)
    sumvalue = sum_value(data[numerical_columns].values)
    return np.column_stack([front_closest_speed_distance, pressure_front, num_cars_front, num_cars_rear, avg_speed, front_distances_all_lanes, rear_distances_all_lanes, lanes_closed_in_front, in_safe_net_front, front_closest_speed, rear_closest_speed, speedgap, maxvalue, minvalue, avgvalue, sumvalue])
