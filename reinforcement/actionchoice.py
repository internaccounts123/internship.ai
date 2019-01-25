import numpy as np


class ActionChoice:

    def __init__(self, explore_start, explore_stop, decay_rate, decay_step):
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.decay_step = decay_step

    def get_exploration_prob(self):
        explore_prob = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(
            -self.decay_rate * self.decay_step)
        return explore_prob

    def get_mode(self, increment_decay=True):
        explore_prob = self.get_exploration_prob()
        mode = np.random.choice(np.array(["explore", "exploit"]), p=[explore_prob, 1.0 - explore_prob])
        if increment_decay:
            self.decay_step += 1
        return mode
