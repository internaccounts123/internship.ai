import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, buffer_size, state_shape):
        self.buffer_size = buffer_size
        self.items_present = 0
        self.curr_index = 0
        self.state_buffer = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.next_state_buffer = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.reward_buffer = np.zeros((buffer_size), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size), dtype=np.int32)

    def add(self, experience):
        state, action, reward, next_state = experience
        self.state_buffer[self.curr_index] = state
        self.action_buffer[self.curr_index] = action
        self.reward_buffer[self.curr_index] = reward
        self.next_state_buffer[self.curr_index] = next_state
        self.curr_index = (self.curr_index + 1) % self.buffer_size
        self.items_present = min(self.items_present + 1, self.buffer_size)

    def sample(self, batch_size):
        inds = np.random.choice(np.arange(self.items_present),
                                size=batch_size,
                                replace=False)

        return self.state_buffer[inds], self.action_buffer[inds], \
            self.reward_buffer[inds], self.next_state_buffer[inds]