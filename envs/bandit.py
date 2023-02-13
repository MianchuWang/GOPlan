import numpy as np

from scipy import stats

class Bandit:

    def __init__(self):
        self.ac_dim = 1
        self.state_dim = 2 # It is a placeholder
        self.goal_dim = 2 # It is a placeholder
        self._max_episode_steps = 1

        self.mus = [-0.5, 0.5]
        self.sigmas = [0.2, 0.2]
        self.normals = [stats.norm(self.mus[0], self.sigmas[1]), stats.norm(self.mus[1], self.sigmas[1])]
        self.weights = [0.5, 0.5]


    def reset(self, seed):
        return self.get_obs(), None

    def step(self, a):
        reward = self.compute_reward(a)
        return self.get_obs(), reward, True, True, None

    def compute_reward(self, a):
        reward = 0
        for i, nor in enumerate(self.normals):
            reward += self.weights[i] * nor.pdf(a)
        return reward

    def get_obs(self):
        return {'observation': np.array([0, 0]),
                'achieved_goal': np.array([0, 0]),
                'desired_goal': np.array([0, 0])}

