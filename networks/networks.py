import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Dynamics(nn.Module):
    def __init__(self, state_dim, ac_dim):
        super(Dynamics, self).__init__()
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.model = nn.Sequential(nn.Linear(state_dim + ac_dim, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, state_dim))

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
        self.model = nn.Sequential(nn.Linear(state_dim + goal_dim, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, ac_dim),
                                   )
        self.log_std = nn.Parameter(torch.zeros(ac_dim), requires_grad=True)

    def forward(self, s, g):
        input = torch.cat((s, g), dim=1)
        loc = torch.tanh(self.model(input))
        scale = torch.exp(self.log_std)
        normal_dist = Normal(loc, scale)
        return normal_dist, loc