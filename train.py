import argparse
import random
import torch
import numpy as np
import wandb

from tqdm import tqdm

from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='FetchReach-v3')
parser.add_argument('--dataset', type=str, default='datasets/gym/expert/FetchReach')
parser.add_argument('--agent', type=str, default='ago')
parser.add_argument('--buffer_capacity', type=int, default=400_0000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=bool, default=True)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=200)

parser.add_argument('--enable_wandb', type=bool, default=True)
parser.add_argument('--pretrain_steps', type=int, default=100000)
parser.add_argument('--eval_episodes', type=int, default=200)
args = parser.parse_args()
print(args)

if args.enable_wandb:
    wandb.init(project='AGO')
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

for i in tqdm(range(args.pretrain_steps)):
    training_info = agent.train_models()
    if args.enable_wandb:
        wandb.log(training_info)

returns = []
for i in tqdm(range(args.eval_episodes)):
    agent.reset()
    obs = env.reset(seed=np.random.randint(1e10))[0]
    for step in range(env_info['max_steps']):
        action = agent.get_action(obs['observation'], obs['desired_goal'])
        obs, reward, _, _, info = env.step(action)
        returns.append(reward)

mean_return = np.array(returns).sum() / args.eval_episodes
print('The return is', mean_return)
