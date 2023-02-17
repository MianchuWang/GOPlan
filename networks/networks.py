import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.distributions.normal import Normal


class Generator(nn.Module):
    def __init__(self, state_dim, ac_dim, goal_dim, noise_dim):
        super().__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.noise_dim = noise_dim
        self.input_dim = self.state_dim + self.goal_dim + noise_dim
        self.model = nn.Sequential(nn.Linear(self.input_dim, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.BatchNorm1d(512),
                                   nn.ReLU(),
                                   nn.Linear(512, self.ac_dim))

    def forward(self, state, goal, noise):
        input = torch.cat([state, goal, noise], dim=-1)
        output = self.model(input)
        return torch.tanh(output)


class Discriminator(nn.Module):
    def __init__(self, state_dim, ac_dim, goal_dim):
        super().__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.goal_dim = goal_dim
        self.input_dim = self.state_dim + self.ac_dim + self.goal_dim
        self.model = nn.Sequential(spectral_norm(nn.Linear(self.input_dim, 512)),
                                   nn.LeakyReLU(),
                                   spectral_norm(nn.Linear(512, 512)),
                                   nn.LeakyReLU(),
                                   spectral_norm(nn.Linear(512, 1)))
    def forward(self, states, actions, goals):
        input = torch.cat([states, actions, goals], dim=-1)
        return self.model(input)


class Dynamics(nn.Module):
    def __init__(self, state_dim, ac_dim):
        super(Dynamics, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.model = nn.Sequential(nn.Linear(state_dim + ac_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, state_dim))

    def forward(self, state, action):
        input = torch.cat([state, action], dim=1)
        output = self.model(input)
        return state + output

class Policy(nn.Module):
    def __init__(self, state_dim, ac_dim, goal_dim):
        super(Policy, self).__init__()
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.ac_dim = ac_dim
        self.model = nn.Sequential(nn.Linear(state_dim + goal_dim, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, 256),
                                   nn.ReLU(),
                                   nn.Linear(256, ac_dim),
                                   )
        self.log_std = nn.Parameter(torch.zeros(ac_dim), requires_grad=True)

    def forward(self, s, g):
        input = torch.cat((s, g), dim=-1)
        loc = torch.tanh(self.model(input))
        scale = torch.exp(self.log_std)
        normal_dist = Normal(loc, scale)
        return normal_dist, loc

class v_network(nn.Module):
    def __init__(self, state_dim, goal_dim):
        super(v_network, self).__init__()
        self.state_dim = state_dim
        self.model = nn.Sequential(nn.Linear(state_dim + goal_dim, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 1))
    def forward(self, state, goal):
        input = torch.cat([state, goal], dim=1)
        output = self.model(input)
        return output
