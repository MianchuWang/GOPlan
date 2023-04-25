import torch
import numpy as np
from agents.base_agent import BaseAgent
from networks.networks import Policy, Contrastive

class CRL(BaseAgent):
    def __init__(self, **agent_params):
        super().__init__(**agent_params)
        self.her_prob = 1.0
        self.policy = Policy(self.state_dim, self.ac_dim, self.goal_dim).to(device=self.device)
        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        self.contrastive = Contrastive(self.state_dim, self.ac_dim, self.goal_dim).to(device=self.device)
        self.contrastive_opt = torch.optim.Adam(self.contrastive.parameters(), lr=3e-4)
    
    def train_models(self):
        self.train_contrastive(batch_size=1024)
        self.train_policy(batch_size=256)
        
    def train_contrastive(self, batch_size):
        I = torch.eye(batch_size).to(self.device)
        states, actions, _, goals = self.replay_buffer.sample(batch_size, her_prob=self.her_prob)
        states, actions, _, goals = self.preprocess(states=states, actions=actions, goals=goals)
        
        sa_repr = self.contrastive.encode_anchor(states, actions)
        g_repr = self.contrastive.encode_target(goals) 
        logits = torch.einsum('ik, jk -> ij', sa_repr, g_repr)
        loss = self.sigmoid_binary_cross_entropy(logits, I).mean()

        self.contrastive_opt.zero_grad()
        loss.backward()
        self.contrastive_opt.step()
        
        accuracy = torch.mean((torch.argmax(logits, dim=1) == torch.argmax(I, dim=1)).to(torch.float32))
        logits_pos = torch.sum(logits * I) / torch.sum(I)
        logits_neg = torch.sum(logits * (1 - I)) / torch.sum(1 - I)
        binary_accuracy = torch.mean(((logits > 0) == I).to(torch.float32))
        return {'contrastive_loss': loss.item(), 
                'categorical_accuracy': accuracy.item(),
                'logits_pos': logits_pos.item(),
                'logits_neg': logits_neg.item(),
                'binary_accuracy': binary_accuracy}
    
    def train_policy(self, batch_size):
        states, actions, _, goals = self.replay_buffer.sample(batch_size, her_prob=self.her_prob)
        states, actions, _, goals = self.preprocess(states=states, actions=actions, goals=goals)
        
        ac_dist, gen_actions = self.policy(states, goals)
        sa_repr = self.contrastive.encode_anchor(states, gen_actions)
        g_repr = self.contrastive.encode_target(goals)
        diag = torch.einsum('ik, ik -> i', sa_repr, g_repr)
        loss_contrastive = - diag.mean()
        
        loss_sl = - ac_dist.log_prob(actions).mean()
        
        loss = 0.25 * loss_contrastive + 0.75 * loss_sl
        self.policy_opt.zero_grad()
        loss.backward()
        self.policy_opt.step()
        return {'policy_loss': loss.item(), 
                'policy_sl_loss': loss_sl.item(),
                'policy_contrastive_loss': loss_contrastive.item()}
    
    def get_action(self, state, goal):
        with torch.no_grad():
            state_prep, _, _, goal_prep = \
                self.preprocess(states=state[np.newaxis], goals=goal[np.newaxis])
            ac_dist, actions_tensor = self.policy(state_prep, goal_prep)
        return actions_tensor.cpu().numpy().squeeze()
    
    def sigmoid_binary_cross_entropy(self, logits, labels):
        log_p = torch.nn.functional.logsigmoid(logits)
        log_not_p = torch.nn.functional.logsigmoid(-logits)
        return - (labels * log_p + (1. - labels) * log_not_p)