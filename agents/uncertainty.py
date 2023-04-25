import torch
import numpy as np

from agents.gcsl import GCSL
from networks.networks import Policy, v_network, AE, q_network

class Uncertainty(GCSL):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.K = 1
        self.latent_dim = 32
        self.her_prob = 1.0
        
        self.autoencoder = AE(self.state_dim, self.goal_dim, latent_dim=self.latent_dim).to(device=self.device)
        self.ae_opt = torch.optim.Adam(self.autoencoder.parameters(), lr=5e-4)
        self.ae_v_net = v_network(self.latent_dim, 0).to(device=self.device)
        self.ae_v_target = v_network(self.latent_dim, 0).to(device=self.device)
        self.ae_v_target.load_state_dict(self.ae_v_net.state_dict())
        self.ae_v_opt = torch.optim.Adam(self.ae_v_net.parameters(), lr=1e-4)
        self.ae_v_training_steps = 0
        
        self.v_net = v_network(self.state_dim, self.goal_dim).to(device=self.device)
        self.v_target_net = v_network(self.state_dim, self.goal_dim).to(device=self.device)
        self.v_target_net.load_state_dict(self.v_net.state_dict())
        self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=5e-4)
        self.v_training_steps = 0

    def train_models(self, batch_size=512):
        value_info = self.train_value_function(batch_size=batch_size)
        policy_info = self.train_policy(batch_size=batch_size)
        ae_info = self.train_autoencoder(batch_size=batch_size)
        ae_value_info = self.train_autoencoder_value_function(batch_size=batch_size)
        if self.v_training_steps % 40 == 0:
            self.update_target_nets(self.v_net, self.v_target_net)
        if self.ae_v_training_steps % 40 == 0:
            self.update_target_nets(self.ae_v_net, self.ae_v_target)
        return {**value_info, **policy_info, **ae_info, **ae_value_info}

    def train_autoencoder_value_function(self, batch_size):
        states, actions, next_states, goals = self.sample_func(batch_size, her_prob=self.her_prob)
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states_prep, _, next_states_prep, goals_prep = \
            self.preprocess(states=states, next_states=next_states, goals=goals)
        with torch.no_grad():
            #v_next_value = self.v_target_net(next_states_prep, goals_prep)
            next_latent = self.autoencoder.encode(next_states_prep, goals_prep)
            v_next_value = self.ae_v_target(next_latent, torch.zeros(batch_size, 0).to(device=self.device))
            target_v_value = rewards_tensor + self.discount * v_next_value
        latent = self.autoencoder.encode(states_prep, goals_prep)
        pred_v_value = self.ae_v_net(latent, torch.zeros(batch_size, 0).to(device=self.device))
        v_loss = ((target_v_value - pred_v_value) ** 2).mean()
        self.ae_v_opt.zero_grad()
        v_loss.backward()
        self.ae_v_opt.step()

        self.ae_v_training_steps += 1
        return {'ae_v_loss': v_loss.item(),
                'ae_pred_value': pred_v_value.mean().item(),
                'ae_target_value': target_v_value.mean().item()}

    def train_autoencoder(self, batch_size):
        states, actions, next_states, goals = self.sample_func(batch_size=batch_size, her_prob=self.her_prob)
        states_prep, _, _, goals_prep = self.preprocess(states=states, goals=goals)
        recon = self.autoencoder(states_prep, goals_prep)
        sg = torch.cat([states_prep, goals_prep], dim=-1)
        recon_loss = torch.nn.functional.mse_loss(recon, sg)

        shuffle_seed = torch.randperm(batch_size)
        # latent distance
        latent = self.autoencoder.encode(states_prep, goals_prep)
        shuffled_latent = latent[shuffle_seed]
        latent_distance = ((latent - shuffled_latent.detach()) ** 2).mean(dim=1)
        # value mse
        values = self.v_net(states_prep, goals_prep)
        shuffled_values = values[shuffle_seed]
        value_distance = ((values - shuffled_values) ** 2).mean(dim=1)
        
        latent_loss = ((latent_distance - value_distance.detach()) ** 2).mean()
        loss = recon_loss + 2000 * latent_loss
        self.ae_opt.zero_grad()
        loss.backward()
        self.ae_opt.step()
        return {'autoencoder': loss.item(),
                'latent_loss': latent_loss.item(),
                'recon_loss': recon_loss.item()}


    def train_value_function(self, batch_size):
        states, actions, next_states, goals = self.sample_func(batch_size, her_prob=self.her_prob)
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
        states, actions, next_states, goals = self.sample_func(batch_size, her_prob=self.her_prob)
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis] 
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states_prep, actions_prep, next_states_prep, goals_prep = \
            self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
        with torch.no_grad():
            
            latent = self.autoencoder.encode(states_prep, goals_prep)
            next_latent = self.autoencoder.encode(next_states_prep, goals_prep)
            value = self.ae_v_net(latent, torch.zeros(batch_size, 0).to(device=self.device))
            next_value = self.ae_v_net(next_latent, torch.zeros(batch_size, 0).to(device=self.device))
            #value = self.v_net(states_prep, goals_prep)
            #next_value = self.v_net(next_states_prep, goals_prep)
            A = rewards + self.discount * next_value - value
            weights = torch.clamp(torch.exp(A), 0, 10)

        _, gen_actions = self.policy(states_prep, goals_prep)
        loss = (((gen_actions - actions_prep) ** 2).mean(dim=1) * weights.squeeze()).mean()
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()

        return {'policy_loss': loss.item()}
        '''
        states, actions, next_states, goals = self.sample_func(batch_size)
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states_prep, actions_prep, next_states_prep, goals_prep = \
            self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
        
        with torch.no_grad():
            #latent = self.autoencoder.encode(states_prep, goals_prep)
            #next_latent = self.autoencoder.encode(next_states_prep, goals_prep)
            #value = self.ae_v_net(latent, torch.zeros(batch_size, 0).to(device=self.device))
            #next_value = self.ae_v_net(next_latent, torch.zeros(batch_size, 0).to(device=self.device))
            value = self.v_nets[0](states_prep, goals_prep)
            next_value = self.v_nets[0](next_states_prep, goals_prep)
            A = rewards + self.discount * next_value - value
            weights = torch.clamp(torch.exp(20 * A + 20), 0, 10)
        
        weights = torch.ones_like(weights)
        _, gen_actions = self.policy(states_prep, goals_prep)
        loss = (((gen_actions - actions_prep) ** 2).mean(dim=1) * weights.squeeze()).mean()
        
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()

        return {'policy_loss': loss.item()}
        '''
