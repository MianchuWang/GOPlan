
import torch
import joblib
import numpy as np

from agents.base_agent import BaseAgent
from networks.networks import Generator, Discriminator, v_network
from dynamics.dynamics import DynamicsModel

class AGO(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)

        self.noise_dim = 16

        self.dynamics = DynamicsModel(3, self.state_dim, self.ac_dim, self.device)

        self.generator = Generator(self.state_dim, self.ac_dim, self.goal_dim, self.noise_dim).to(device=self.device)
        self.discriminator = Discriminator(self.state_dim, self.ac_dim, self.goal_dim).to(device=self.device)
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.v_net = v_network(self.state_dim, self.goal_dim).to(device=self.device)
        self.v_target_net = v_network(self.state_dim, self.goal_dim).to(device=self.device)
        self.v_target_net.load_state_dict(self.v_net.state_dict())
        self.v_net_opt = torch.optim.Adam(self.v_net.parameters(), lr=1e-3)
        self.v_training_steps = 0

        self.her_prob = 1.0
        self.gan_l2 = 0
        self.v_training_steps = 0

    def train_models(self):
        wgan_info = self.train_wgan(batch_size=128)
        dynamics_info = self.train_dynamics(batch_size=512)
        return {**wgan_info, **dynamics_info}
        '''
        value_function_info = self.train_value_function(batch_size=512)
        if self.v_training_steps % 2 == 0:
            self.update_target_nets(self.v_net, self.v_target_net)
        return {**wgan_info, **value_function_info}
        '''
        #return wgan_info

    def train_wgan(self, batch_size):
        # Real actions -> higher score
        # Fake actions -> lower score

        self.discriminator.train()
        self.generator.train()

        states, actions, next_states, goals = self.replay_buffer.sample(batch_size, her_prob=self.her_prob)
        states_prep, actions_prep, next_states_prep, goals_prep = \
            self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
        '''
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        pred_v_value = self.v_net(states_prep, goals_prep)
        next_v_value = self.v_net(next_states_prep, goals_prep)
        A = rewards_tensor + (1 - rewards_tensor) * self.discount * next_v_value - pred_v_value
        clip_exp_A = torch.clamp(torch.exp(A), 0, 10)
        weights = torch.softmax(clip_exp_A + clip_exp_A.mean(), dim=0)
        '''
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_actions = self.generator(states_prep, goals_prep, noise)
        dis_loss_fake = self.discriminator(states_prep, fake_actions, goals_prep).mean()
        #dis_loss_real = (weights * self.discriminator(states_prep, actions_prep, goals_prep)).sum()
        dis_loss_real = self.discriminator(states_prep, actions_prep, goals_prep).mean()
        dis_main_loss = dis_loss_fake - dis_loss_real
        dis_reg = 0
        for par in self.discriminator.parameters():
            dis_reg += torch.dot(par.view(-1), par.view(-1))
        dis_loss = dis_main_loss + self.gan_l2 * dis_reg
        self.discriminator_opt.zero_grad()
        dis_loss.backward()
        self.discriminator_opt.step()

        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake_actions = self.generator(states_prep, goals_prep, noise)
        gen_main_loss = - self.discriminator(states_prep, fake_actions, goals_prep).mean()
        gen_reg = 0
        for par in self.generator.parameters():
            gen_reg += torch.dot(par.view(-1), par.view(-1))
        gen_loss = gen_main_loss + self.gan_l2 * gen_reg
        self.generator_opt.zero_grad()
        gen_loss.backward()
        self.generator_opt.step()

        self.generator.eval()
        self.discriminator.eval()
        return {#'advantages': A,
                #'clip_exp_A': clip_exp_A,
                #'weights': weights,
                'generator_loss': gen_loss.item(),
                'discriminator_loss': dis_loss.item()}

    def train_value_function(self, batch_size):
        states, actions, next_states, goals = self.replay_buffer.sample(batch_size, self.her_prob)
        achieved_goals = self.get_goal_from_state(next_states)
        rewards = self.compute_reward(achieved_goals, goals, None)[..., np.newaxis]
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        states_prep, actions_prep, next_states_prep, goals_prep = \
            self.preprocess(states=states, actions=actions, next_states=next_states, goals=goals)
        with torch.no_grad():
            v_next_value = self.v_target_net(next_states_prep, goals_prep)
            target_v_value = rewards_tensor + (1-rewards_tensor) * self.discount * v_next_value
        pred_v_value = self.v_net(states_prep, goals_prep)
        v_loss = ((target_v_value - pred_v_value) ** 2).mean()
        self.v_net_opt.zero_grad()
        v_loss.backward()
        self.v_net_opt.step()
        self.v_training_steps += 1
        return {'v_loss': v_loss.item(),
                'v_value': pred_v_value.mean().item()}

    def update_target_nets(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)

    def train_dynamics(self, batch_size):
        states, actions, next_states, _ = self.replay_buffer.sample(batch_size, self.her_prob)
        states_prep, actions_prep, next_states_prep, _ = \
            self.preprocess(states=states, actions=actions, next_states=next_states)
        dynamics_info = self.dynamics.train_models(states_prep, actions_prep, next_states_prep)
        return dynamics_info

    def get_action(self, state, goal):
        with torch.no_grad():
            state_prep, _, _, goal_prep = self.preprocess(states=state[np.newaxis], goals=goal[np.newaxis])
            noise = torch.zeros(1, self.noise_dim, device=self.device)
            action = self.generator(state_prep, goal_prep, noise)
        return action.cpu().numpy().squeeze()

    def plan(self, state, goal):
        num_acs = 25
        num_copies = 100
        max_steps = 20

        with torch.no_grad():
            states_prep, _, _, goals_prep = self.preprocess(states=state[np.newaxis].repeat(num_acs, axis=0),
                                                            goals=goal[np.newaxis].repeat(num_acs, axis=0))

            noise = torch.randn(num_acs, self.noise_dim, device=self.device)
            gen_actions = self.generator(states_prep, goals_prep, noise)
            pred_next_states = self.dynamics.predict(states_prep, gen_actions)

            rep_goals = goals_prep.repeat(num_copies, 1)
            all_states = torch.zeros(num_acs * num_copies, max_steps + 1, self.state_dim, device=self.device)
            all_states[:, 0] = pred_next_states.repeat(num_copies, 1)

            for h in range(0, max_steps):
                noise = torch.randn(num_acs * num_copies, self.noise_dim, device=self.device)
                actions = self.generator(all_states[:, h], rep_goals, noise)
                all_states[:, h + 1] = self.dynamics.predict(all_states[:, h], actions)

            unnormalised_states, _, _, _ = self.postprocess(states=all_states.reshape(-1, self.state_dim))
            achieved_goals = self.get_goal_from_state(unnormalised_states)

            rewards = self.compute_reward(achieved_goals,
                                          goal[np.newaxis].repeat(num_acs * num_copies * (max_steps + 1), axis=0),
                                          None)
            rewards = rewards.reshape(num_copies, num_acs, max_steps + 1)

            # discount
            '''
            gammas = np.ones(max_steps + 1)
            for t in range(max_steps + 1):
                gammas[t] = gammas[t] * (self.discount ** t)
            rewards = rewards * gammas
            '''
            rewards = rewards.sum(axis=2).mean(axis=0)

            kappa = 2
            gen_actions = gen_actions.cpu().numpy()
            action = (gen_actions * np.exp(kappa * rewards[..., np.newaxis])).sum(axis=0) / \
                     np.exp(kappa * rewards[..., np.newaxis]).sum()

            return action



