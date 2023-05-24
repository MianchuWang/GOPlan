import torch
import numpy as np

from agents.gcsl import GCSL
from networks.networks import Policy, v_network

class GEAW(GCSL):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.v_net = v_network(self.state_dim, self.goal_dim).to(device=self.device)
        self.v_target_net = v_network(self.state_dim, self.goal_dim).to(device=self.device)
        self.v_target_net.load_state_dict(self.v_net.state_dict())
        self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=5e-4)
        self.v_training_steps = 0

    def train_models(self, batch_size=512):
        value_info = self.train_value_function(batch_size=512)
        policy_info = self.train_policy(batch_size=512)
        if self.v_training_steps % 2 == 0:
            self.update_target_nets(self.v_net, self.v_target_net)
        return {**value_info, **policy_info}

    def train_value_function(self, batch_size):
        states, actions, next_states, goals = self.sample_func(batch_size)
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states_prep, _, next_states_prep, goals_prep = \
            self.preprocess(states=states, next_states=next_states, goals=goals)
        with torch.no_grad():
            v_next_value = self.v_target_net(next_states_prep, goals_prep)
            target_v_value = rewards_tensor + self.discount * v_next_value
        pred_v_value = self.v_net(states_prep, goals_prep)
        v_loss = ((target_v_value - pred_v_value) ** 2).mean()
        self.v_net_opt.zero_grad()
        v_loss.backward()
        self.v_net_opt.step()

        self.v_training_steps += 1
        return {'v_loss': v_loss.item(),
                'pred_value': pred_v_value.mean().item(),
                'target_value': target_v_value.mean().item()}

    def update_target_nets(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(0.05 * param.data + 0.95 * target_param.data)

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

        _, gen_actions = self.policy(states_prep, goals_prep)
        loss = (((gen_actions - actions_prep) ** 2).mean(dim=1) * weights.squeeze()).mean()
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()

        return {'policy_loss': loss.item(), 'training reward': rewards.mean().item()}

    def save(self, path):
        models = (self.policy, self.v_net, self.v_target_net)
        import joblib
        joblib.dump(models, path)
