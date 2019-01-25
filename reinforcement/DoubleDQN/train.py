import numpy as np
import keras.optimizers as KO
import sys
sys.path.append("../../") ####ROODIR HERE
from reinforcement.env_tools\
    import reward_calculator,preprocess_state
from reinforcement.network import Network
from reinforcement.actionchoice import ActionChoice
from reinforcement.buffer import ExperienceReplayBuffer


class DeepQLearning:
    """
    This class implements the Double Deep Q learning
    """
    def __init__(self, network_file_path, encoders_paths):
        """

        :param network_file_path: Path of file that is used to construct the network (specifies layers and units)
        :param encoders_paths: Path of label encoders
        This constructor sets hyperparameters values
        and makes networks
        """
        
        self.encoders_path = encoders_paths

        # env variables
        self.obs_shape = [471]
        self.num_actions = 5
        self.action_ind_mapper = np.array(["Accelerate", "Decelerate", "Keep", "LeftLaneChange", "RightLaneChange"])
        
        # experience replay and its parameters
        self._base_path = '../ai.aai/ai.aai.intelligence/reinforcement/internship'
        self.memory_size = 10000
        self.memory = ExperienceReplayBuffer(self.memory_size, self.obs_shape)

        # parameters for choosing action
        self.prob_explore_start = 1.0            # exploration probability at start
        self.prob_explore_stop = 0.01            # minimum exploration probability 
        self.prob_decay_rate = 0.001
        self.action_stategy = ActionChoice(explore_start=self.prob_explore_start,
                                           explore_stop=self.prob_explore_stop,
                                           decay_rate=self.prob_decay_rate,
                                           decay_step=0)
        # training params
        self.num_pretrain_frames = 6000
        self.gamma = 0.99
        self.curr_frame_no = 0
        self.epoch_no = 0
        self.dqn_update_freq = 4
        self.tn_update_freq = 100
        self.buffer_sample_size = 128
        self.batch_size = 128
        self.print_every_n_epoch = 100
        self.save_every_n_epochs=10

        self.prev_state = None
        self.prev_action = 1
        self.learning_rate = 0.0005
        self.design_config_file = network_file_path

        self.num_frames_skip = 10
        # forming networks

        optimizer = KO.Adam(lr=self.learning_rate)
        self.main_network = Network(input_shape=self.obs_shape,num_outputs=self.num_actions,
                                    optimizer=optimizer,arch_config_file=self.design_config_file)
        self.main_network.construct_model()

        optimizer = KO.Adam(lr=self.learning_rate)
        self.target_network = Network(input_shape=self.obs_shape, num_outputs=self.num_actions,
                                      optimizer =optimizer, arch_config_file=self.design_config_file)
        self.target_network.construct_model()

    def choose_action(self, state):
        """
        This func chooses an action based on the state/obs received and stores it on the buffer and calls the
        training function
        :param state: Raw unprocessed state
        :return: action (string type)

        """
        # choose action using egreedy approach
        # returns the choosen action index
        
        mode = self.action_stategy.get_mode(increment_decay=True)
        
        action = None
        if mode == "explore":
            action = np.random.choice(np.arange(self.num_actions))

        elif mode == "exploit":
            qvals = self.main_network.predict(np.expand_dims(state,axis=0))
            action = (np.argmax(qvals,axis=-1))[0]

        else:
            raise Exception("invalid mode value received. Valid vals include explore,exploit")
        return action

    def train_predict(self, raw_state):
        if self.curr_frame_no%self.num_frames_skip:
            self.curr_frame_no += 1
            return self.action_ind_mapper[self.prev_action]

        state = preprocess_state(self.encoders_path, raw_state)
        action = self.choose_action(state)
        mapped_action = self.action_ind_mapper[action]

        # append to buffer
        if self.curr_frame_no != 0:
            # find reward
            reward = reward_calculator(state, mapped_action)
            print("Reward {}".format(reward))
            self.memory.add((self.prev_state, self.prev_action, reward, state))
                                         
        #train networks
        self.train_networks()
        
        self.prev_state = state
        self.prev_action = action
        self.curr_frame_no += 1
                                         
        return mapped_action
    
    def train_networks(self):
        """
        trains the main networks afer every few frames
        updates target networks using the weights of the main network
        saves weights
        :return:
        """
        
        if self.curr_frame_no<self.num_pretrain_frames:
            return
        elif self.curr_frame_no==self.num_pretrain_frames:
            print("===========>Buffer filled<=======================\n\n")
            print(self.memory.items_present, "\n")
        
        if self.curr_frame_no % self.dqn_update_freq == 0:
            # Obtain random mini-batch from memory
            # Obtain random mini-batch from memory
            states_mb, actions_mb, rewards_mb, next_states_mb = self.memory.sample(self.buffer_sample_size)

            # Get Q values for next_state 
            qs_next_state_mb = self.target_network.predict(next_states_mb)
            actions_next_state_mb = self.main_network.predict(next_states_mb)
            actions_next_state_mb = np.argmax(actions_next_state_mb, axis=-1)
            target_qs_next_state_mb = qs_next_state_mb[np.arange(self.buffer_sample_size), actions_next_state_mb]
            targets_qvals_mb = rewards_mb+(self.gamma*target_qs_next_state_mb)


            #converting actions to one hot
            conv_mat=np.eye(self.num_actions)
            actions_one_hot_mb = conv_mat[actions_mb]
                                         
            # if self.epoch_no%self.print_every_n_epoch == 0:
            #     print("Epoch {} ".format(self.epoch_no))

            print("Executing training step")
            self.main_network.Model.fit(
              x=[states_mb, actions_one_hot_mb],
              y=[targets_qvals_mb],
              epochs=1,
              batch_size=self.batch_size,
              verbose=1
            )
            self.epoch_no += 1

            if self.epoch_no % self.save_every_n_epochs == 0:
                self.main_network.Model.save_weights("model_weights.h5")
        
        if self.curr_frame_no % self.tn_update_freq == 0:
            print("=======updating target's weights=======")
            self.target_network.Model.set_weights(self.main_network.Model.get_weights())
        




