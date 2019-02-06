import pickle
import os


def reward_calculator(observation, action):
    """
    calculates reward given an observation and action taken for that obs
    :param observation: 1D obs
    :param action: action taken for that obs
    :return: reward
    """
    reward = 0
    optimal_speed = observation[-3]  # Speedlimit - 2
    speed = observation[-4]
    safe_net_front = observation[-5]
    car_start_index = 314   # our cars starting index in obnet
    car_front_index = car_start_index - 15
    reward = reward - abs(speed-optimal_speed)

    # if we have invalid location on our car's front
    if  observation[car_front_index] == -10:
        reward = reward - 500
        return reward

    safe = car_front_index
    for i in range(int(safe_net_front)):
        if observation[safe] != -1:
            reward=reward - 500
            return reward
        safe = safe-3

    # if we are on left lane and our action is leftlanechange
    if (observation[car_start_index-1] == observation[car_start_index-4] == observation[car_start_index-7] == -10)\
            and action == 'LeftLaneChange':
        reward = reward - 500
        return reward

    if (observation[car_start_index+1] == observation[car_start_index+4] == observation[car_start_index+7] == -10)\
            and action == 'RightLaneChange':
        reward = reward - 500
        return reward
    # if our car's blocks have overlapping values collision has occured
    if observation[car_start_index] == observation[car_start_index - 3] == observation[car_start_index - 6]\
            == observation[car_start_index - 9] == observation[car_start_index - 12]:
        reward += 0
    else:
        reward = reward - 500
        return reward
    return reward


def preprocess_state(path, state):
    """

    :param path: path of encoders
    :param state: 1D obs
    :return:
    """
    state = state[1:]
    with open(os.path.join(path, 'lane_change_mode.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
        state[0] = str(label_encoder.transform([state[0]])[0])
    with open(os.path.join(path, 'previous_decision.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
        state[-2] = str(label_encoder.transform([state[-2]])[0])
    state = state.astype(float)
    return state
