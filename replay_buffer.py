import numpy as np
import joblib

class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, ac_dim, goal_dim, max_steps, get_goal_from_state):

        self.entries = int(buffer_size / max_steps)
        self.obs = np.zeros((self.entries, max_steps, state_dim), dtype=np.float64)
        self.next_obs = np.zeros((self.entries, max_steps, state_dim), dtype=np.float64)
        self.actions = np.zeros((self.entries, max_steps, ac_dim))
        self.goals = np.zeros((self.entries, max_steps, goal_dim), dtype=np.float64)
        self.traj_len = np.zeros((self.entries), dtype=np.int32)

        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.ac_dim = ac_dim
        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.get_goal_from_state = get_goal_from_state

        self.is_full = False
        self.curr_ptr = 0

    def push(self, input):
        traj_len = input['observations'].shape[0]
        self.obs[self.curr_ptr, :traj_len] = input['observations']
        self.next_obs[self.curr_ptr, :traj_len] = input['next_observations']
        self.actions[self.curr_ptr, :traj_len] = input['actions']
        self.goals[self.curr_ptr, :traj_len] = input['desired_goals']
        self.traj_len[self.curr_ptr] = traj_len

        self.curr_ptr += 1
        if self.curr_ptr == self.entries:
            self.curr_ptr = 0
            self.is_full = True

    def load_dataset(self, path):
        with open(path+'.pkl', 'rb') as f:
            loaded_data = joblib.load(f)
        size = loaded_data['o'].shape[0]
        for i in range(size):
            self.push({'observations': loaded_data['o'][i, :-1],
                       'next_observations': loaded_data['o'][i, 1:],
                       'actions': loaded_data['u'][i],
                       'desired_goals': loaded_data['g'][i]})

    def get_bounds(self):
        boarder = self.entries if self.is_full else self.curr_ptr
        return 0, boarder

    def sample_her(self, batch_size, her_prob=0):
        low, high = self.get_bounds()
        entry = np.random.randint(0, high, batch_size)
        index = np.random.randint(0, self.traj_len[entry])
        ret_obs = self.obs[entry, index]
        ret_next_obs = self.next_obs[entry, index]
        ret_actions = self.actions[entry, index]

        # hindsight experience replay
        goal_index = np.random.randint(index, self.traj_len[entry], batch_size)
        replace = (np.random.rand(batch_size) < her_prob).astype(np.int32)
        ret_goals = self.get_goal_from_state(self.next_obs[entry, goal_index]) * replace[..., np.newaxis] + \
                    self.goals[entry, 0] * (1 - replace[..., np.newaxis])
        return ret_obs, ret_actions, ret_next_obs, ret_goals

    def sample_sequence(self, batch_size, tau):
        high = self.get_high()
        entry = np.random.randint(0, high, batch_size)
        index = np.random.randint(0, self.traj_len[entry] - tau, batch_size)
        ret_obs = np.concatenate([self.obs[entry, np.newaxis, index + t] for t in range(tau)], axis=1)
        ret_next_obs = np.concatenate([self.next_obs[entry, np.newaxis, index + t] for t in range(tau)], axis=1)
        ret_actions = np.concatenate([self.actions[entry, np.newaxis, index + t] for t in range(tau)], axis=1)
        goal_index = np.random.randint(index + tau, self.traj_len[entry], batch_size)
        ret_goals = self.get_goal_from_state(self.obs[entry, np.newaxis, goal_index]).repeat(tau, axis=1)

        return_to_go = index + tau - goal_index
        return_to_go = np.concatenate([return_to_go[:, np.newaxis, np.newaxis] - i
                                       for i in range(tau, 0, -1)], axis=1)
        return ret_obs, ret_actions, ret_next_obs, ret_goals, return_to_go