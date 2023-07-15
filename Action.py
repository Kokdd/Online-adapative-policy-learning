import numpy as np


class action(object):

    def __init__(self,t):
        self.pro_score = np.zeros(t + 2)
        self.reward = 0
        self.reward_mean = 0
        self.number = 0

    def update_rm_n(self, observation):
        self.reward_mean = (self.reward_mean * self.number + observation.value) / (self.number + 1)
        self.number += 1
