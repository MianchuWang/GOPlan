import argparse
import random
import torch
import numpy as np
import wandb
import seaborn
import matplotlib.pyplot as plt

from tqdm import tqdm


from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent
from controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='bandit')
parser.add_argument('--dataset', type=str, default='datasets/bandit/bandit')
parser.add_argument('--agent', type=str, default='gcsl')
parser.add_argument('--buffer_capacity', type=int, default=400_0000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=200)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--pretrain_steps', type=int, default=10000)
parser.add_argument('--eval_episodes', type=int, default=100000)
args = parser.parse_args()
print(args)

if args.enable_wandb:
    wandb.init(project='AGO', config=args)
device = "cuda" if torch.cuda.is_available() else "cpu"
env, env_info = return_environment(args.env_name, render_mode=args.render_mode)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

buffer = ReplayBuffer(buffer_size=args.buffer_capacity, state_dim=env_info['state_dim'],
                      ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'],
                      max_steps=env_info['max_steps'], get_goal_from_state=env_info['get_goal_from_state'])
buffer.load_dataset(args.dataset)

agent = return_agent(agent=args.agent, replay_buffer=buffer, state_dim=env_info['state_dim'],
                     ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'], device=device,
                     discount=args.discount, max_steps=env_info['max_steps'], normalise=args.normalise,
                     get_goal_from_state=env_info['get_goal_from_state'],
                     compute_reward=env_info['compute_reward'])

controller = Controller(pretrain_steps=args.pretrain_steps, eval_episodes=args.eval_episodes,
                        enable_wandb=args.enable_wandb, env=env, env_info=env_info,
                        agent=agent, buffer=buffer)

controller.train()
actions = controller.eval()

#seaborn.histplot(buffer.actions[:buffer.curr_ptr].squeeze(), bins=20, binrange=(-1.4, 1.4),
#                 stat='density', color='white')
seaborn.lineplot(x=np.arange(-2, 2, 0.01), y=env.compute_reward(np.arange(-2, 2, 0.01)),
                 linewidth=2, color='black')
seaborn.histplot(actions, bins=20, binrange=(-1.4, 1.4), stat='density', edgecolor='red', color='white')
plt.show()