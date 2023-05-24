import torch
import numpy as np
from agents.geaw import GEAW
from networks.networks import Policy, v_network


class Advque:
    def __init__(self, size=50000):
        self.size = size 
        self.current_size = 0
        self.que = np.zeros(size)
        self.idx = 0
    
    def update(self, values):
        l = len(values)

        if self.idx + l <= self.size:
            idxes = np.arange(self.idx, self.idx+l)
        else:
            idx1 = np.arange(self.idx, self.size)
            idx2 = np.arange(0, self.idx+l -self.size)
            idxes = np.concatenate((idx1, idx2))
        self.que[idxes] = values.reshape(-1)

        self.idx = (self.idx + l) % self.size 
        self.current_size = min(self.current_size+l, self.size)

    def get(self, threshold):
        return np.percentile(self.que[:self.current_size], threshold)



class WGCSL(GEAW):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.adv_que = Advque()
        self.quantile_threshold = 0.0
        self.maximum_thre = 80

    def train_policy(self, batch_size):
        states, actions, next_states, goals = self.sample_func(batch_size)
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis] 
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states_prep, actions_prep, next_states_prep, goals_prep = \
            self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
        with torch.no_grad():
            value = self.v_net(states_prep, goals_prep)
            next_value = self.v_net(next_states_prep, goals_prep)
            A = rewards + self.discount * next_value - value
            weights = torch.clamp(torch.exp(A), 0, 10)

            self.adv_que.update(A.cpu().numpy())
            temp_threshold = self.adv_que.get(self.quantile_threshold)
            positive = torch.ones_like(A)
            positive[A < temp_threshold] = 0.05
            weights *= positive

        _, gen_actions = self.policy(states_prep, goals_prep)
        loss = (((gen_actions - actions_prep) ** 2).mean(dim=1) * weights.squeeze()).mean()
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
        
        self.quantile_threshold = min(self.quantile_threshold + 0.0004, self.maximum_thre)
        return {'policy_loss': loss.item(), 'training reward': rewards.mean().item(),
                'quantile_threshold': self.quantile_threshold}

 