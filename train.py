import argparse
import random
import torch
import numpy as np
import wandb
import os
import time
import seaborn
import matplotlib.pyplot as plt

from tqdm import tqdm


from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent
from controller import Controller

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='SawyerDoor')
parser.add_argument('--dataset', type=str, default='datasets/sawyer/expert/SawyerDoor')
parser.add_argument('--agent', type=str, default='ago')
parser.add_argument('--buffer_capacity', type=int, default=4000000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=100)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--pretrain_steps', type=int, default=500000)
parser.add_argument('--eval_episodes', type=int, default=200)
args = parser.parse_args()
print(args)

if args.enable_wandb:
    wandb.init(project='AGO', config=args)
curr_time = time.gmtime()
experiments_dir = '-'.join(['experiments/' + args.env_name.split('-')[0], str(curr_time.tm_mon), str(curr_time.tm_mday),
                            str(curr_time.tm_hour), str(curr_time.tm_min), str(curr_time.tm_sec) + '/'])
os.makedirs(experiments_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
env, env_info = return_environment(args.env_name, render_mode=args.render_mode)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

buffer = ReplayBuffer(buffer_size=args.buffer_capacity, state_dim=env_info['state_dim'],
                      ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'],
                      max_steps=env_info['max_steps'], get_goal_from_state=env_info['get_goal_from_state'],
                      compute_reward=env_info['compute_reward'])
buffer.load_dataset(args.dataset)

agent = return_agent(agent=args.agent, replay_buffer=buffer, state_dim=env_info['state_dim'],
                     ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'], device=device,
                     discount=args.discount, max_steps=env_info['max_steps'], normalise=args.normalise,
                     get_goal_from_state=env_info['get_goal_from_state'],
                     compute_reward=env_info['compute_reward'])

controller = Controller(pretrain_steps=args.pretrain_steps, eval_episodes=args.eval_episodes,
                        enable_wandb=args.enable_wandb, experiments_dir=experiments_dir, env=env, env_info=env_info,
                        agent=agent, buffer=buffer)

controller.train()
actions = controller.eval()
