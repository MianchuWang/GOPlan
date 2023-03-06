
import torch
import torch.nn as nn
from networks.networks import Dynamics

class DynamicsModel(nn.Module):
    def __init__(self, num_models, state_dim, ac_dim, device):
        super().__init__()
        self.num_models = num_models
        self.state_dim = state_dim
        self.ac_dim = ac_dim
        self.device = device

        self.models = [Dynamics(state_dim, ac_dim).to(device=self.device) for _ in range(num_models)]
        self.opts = [torch.optim.Adam(self.models[i].parameters(), lr=1e-3) for i in range(num_models)]
        self.mse_loss = nn.MSELoss()

    def train_models(self, states, actions, next_states):
        if len(states.shape) == 2:
            for i in range(self.num_models):
                pred_next_states = self.models[i](states, actions)
                loss = self.mse_loss(pred_next_states, next_states)
                self.opts[i].zero_grad()
                loss.backward()
                self.opts[i].step()
        elif len(states.shape) == 3:
            tau = states.shape[1]
            for i in range(self.num_models):
                pred_next_states = torch.zeros_like(next_states)
                curr_states = states[:, 0]
                for t in range(tau):
                    pred_next_states[:, t] = self.models[i](curr_states, actions[:, t])
                    curr_states = pred_next_states[:, t]
                loss = self.mse_loss(pred_next_states, next_states)
                self.opts[i].zero_grad()
                loss.backward()
                self.opts[i].step()
        else:
            raise Exception('Invalid training data.')
        return {'dynamics_loss': loss.item()}

    def uncertainty(self, states, actions):
        length = states.shape[0] - 1
        pred_states = torch.zeros(self.num_models, length, self.state_dim, device=self.device)
        for i in range(length):
            for k in range(self.num_models):
                pred_states[k, i] = self.models[k](states[i].unsqueeze(0), actions[i].unsqueeze(0))
        variances = pred_states.var(0)
        return torch.max(variances.mean(dim=1)).item()



    def predict(self, states, actions, mean=False):
        pred_next_states = torch.zeros(states.shape).to(device=self.device)
        if mean:
            for i in range(self.num_models):
                pred_next_states += self.models[i](states, actions) / self.num_models
        else:
            index = torch.randint(self.num_models, (states.shape[0],)).to(device=self.device)
            for i in range(self.num_models):
                indices = (index==i).to(torch.int32).unsqueeze(dim=-1)
                pred_next_states += self.models[i](states, actions) * indices
        return pred_next_states


