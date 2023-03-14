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
parser.add_argument('--env_name', type=str, default='HandReach-v0')
parser.add_argument('--dataset', type=str, default='datasets/gym/expert/HandReach')
parser.add_argument('--agent', type=str, default='ago')
parser.add_argument('--buffer_capacity', type=int, default=4000000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=300)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--pretrain_steps', type=int, default=500000)
parser.add_argument('--eval_episodes', type=int, default=200)
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
                      max_steps=env_info['max_steps'], get_goal_from_state=env_info['get_goal_from_state'],
                      compute_reward=env_info['compute_reward'])
buffer.load_dataset(args.dataset)

agent = return_agent(agent=args.agent, replay_buffer=buffer, state_dim=env_info['state_dim'],
                     ac_dim=env_info['ac_dim'], goal_dim=env_info['goal_dim'], device=device,
                     discount=args.discount, max_steps=env_info['max_steps'], normalise=args.normalise,
                     get_goal_from_state=env_info['get_goal_from_state'],
                     compute_reward=env_info['compute_reward'])

controller = Controller(pretrain_steps=args.pretrain_steps, eval_episodes=args.eval_episodes,
                        enable_wandb=args.enable_wandb, env=env, env_info=env_info,
                        agent=agent, buffer=buffer)

agent.load(env_info['env_name'] + '-pretrain')

def plot_trajectories():
    agent.reset()
    obs = env.reset(seed=np.random.randint(1e10))[0]

    # Imagined Trajectory
    imagined_states = np.zeros((env_info['max_steps']+1, env_info['state_dim']))
    actions = np.zeros((env_info['max_steps'], env_info['ac_dim']))
    imagined_states[0] = obs['observation']
    for step in range(env_info['max_steps']):
        ac = agent.plan(imagined_states[step], obs['desired_goal'])
        state_prep, ac_prep, _, goal_prep = agent.preprocess(states=imagined_states[step][np.newaxis],
                                                             actions=ac[np.newaxis],
                                                             goals=obs['desired_goal'][np.newaxis])
        ns = agent.dynamics.predict(state_prep, ac_prep, mean=True)
        imagined_states[step + 1], _, _, _ = agent.postprocess(states=ns)
        actions[step] = ac

    real_states = np.zeros((env_info['max_steps']+1, env_info['state_dim']))
    real_states[0] = obs['observation']
    for step in range(env_info['max_steps']):
        obs, _, _, _, _ = env.step(actions[step])
        real_states[step+1] = obs['observation']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    imagined_x_vals = env_info['get_goal_from_state'](imagined_states)[..., 0]
    imagined_y_vals = env_info['get_goal_from_state'](imagined_states)[..., 1]
    imagined_z_vals = env_info['get_goal_from_state'](imagined_states)[..., 2]
    ax.scatter(imagined_x_vals, imagined_y_vals, imagined_z_vals, c='b', marker='o')

    real_x_vals = env_info['get_goal_from_state'](real_states)[..., 0]
    real_y_vals = env_info['get_goal_from_state'](real_states)[..., 1]
    real_z_vals = env_info['get_goal_from_state'](real_states)[..., 2]
    ax.scatter(real_x_vals, real_y_vals, real_z_vals, c='r', marker='o')

    ax.scatter(real_x_vals[0], real_y_vals[0], real_z_vals[0], c='g', marker='P', s=100)
    ax.scatter(obs['desired_goal'][0], obs['desired_goal'][1], obs['desired_goal'][2], c='g', marker='*')

    plt.show()