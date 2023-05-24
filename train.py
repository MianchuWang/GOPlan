import argparse
import random
import torch
import numpy as np
import wandb
import os
import time
import seaborn
import matplotlib.pyplot as plt

from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent
from controller import Controller
import logger

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='FetchPickAndPlace')
parser.add_argument('--dataset', type=str, default='datasets/gym/FetchPickAndPlace')
parser.add_argument('--agent', type=str, default='ago')
parser.add_argument('--buffer_capacity', type=int, default=4000000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=00)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=1)
parser.add_argument('--project', type=str, default='test')
parser.add_argument('--group', type=str, default='test')
parser.add_argument('--pretrain_steps', type=int, default=250000)  
parser.add_argument('--eval_episodes', type=int, default=100) 
parser.add_argument('--eval_every', type=int, default=5000)
parser.add_argument('--log_path', type=str, default='./experiments/')
args = parser.parse_args()
print(args)


if args.enable_wandb:
    wandb.init(project=args.project, config=args, group=args.group, name='{}_{}_seed{}'.format(args.agent,args.env_name, args.seed))
curr_time = time.gmtime()

experiments_dir = args.log_path + args.project + '/' + args.group + '/' + '{}_{}_seed{}'.format(args.agent,args.env_name, args.seed) + '/'

logger.configure(experiments_dir)
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
if args.env_name.startswith('antmaze'):
    buffer.load_d4rl(env)
else:
    buffer.load_dataset(args.dataset)

agent = return_agent(agent=args.agent, replay_buffer=buffer, state_dim=env_info['state_dim'],
                     ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'], device=device,
                     discount=args.discount, max_steps=env_info['max_steps'], normalise=args.normalise,
                     get_goal_from_state=env_info['get_goal_from_state'],
                     compute_reward=env_info['compute_reward'])

controller = Controller(pretrain_steps=args.pretrain_steps, eval_episodes=args.eval_episodes, eval_every=args.eval_every,
                        enable_wandb=args.enable_wandb, experiments_dir=experiments_dir, env=env, env_info=env_info,
                        agent=agent, buffer=buffer)

controller.train()
### evaluation
if 'Push' in args.env_name:
    env_lis = ['FetchPushOOD-Right2Right-v1', 'FetchPushOOD-Left2Left-v1',
                'FetchPushOOD-Left2Right-v1', 'FetchPushOOD-Right2Left-v1']
else:
    env_lis = ['FetchPickOOD-Right2Right-v1', 'FetchPickOOD-Right2Left-v1',
                'FetchPickOOD-Left2Left-v1', 'FetchPickOOD-Left2Right-v1']

for env_name in env_lis:
    logger.log('ENV: {}'.format(env_name))
    eval_env, _ = return_environment(args.env_name, render_mode=args.render_mode)
    
    res = controller.eval_env(eval_env, mode='action', episodes=200)
    for key, val in res.items():
        logger.log('policy evaluation {}: {}'.format(key, val))

    if args.agent == 'ago':
        res = controller.eval_env(eval_env, mode='plan', episodes=200)
        controller.agent.save(experiments_dir + 'finetune_model')
    else:
        res = {'plan': 0}

    for key, val in res.items():
        logger.log('plan evaluation {}: {}'.format(key, val))



