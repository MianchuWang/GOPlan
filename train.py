import argparse
import random
import torch
import numpy as np
import wandb
import os
import time
from tqdm import tqdm

from replay_buffer import ReplayBuffer
from envs import return_environment
from agents import return_agent
import logger

np.seterr(all='ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='FetchPush-v1')
parser.add_argument('--dataset', type=str, default='datasets/FetchPush')
parser.add_argument('--agent', type=str, default='GOPlan')
parser.add_argument('--buffer_capacity', type=int, default=4000000)
parser.add_argument('--discount', type=float, default=0.98)
parser.add_argument('--normalise', type=int, choices=[0, 1], default=1)
parser.add_argument('--render_mode', type=str, default=None)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--pretrain', type=int, choices=[0, 1], default=0)

parser.add_argument('--enable_wandb', type=int, choices=[0, 1], default=0)
parser.add_argument('--project', type=str, default='gymrobotics')
parser.add_argument('--group', type=str, default='')
parser.add_argument('--pretrain_steps', type=int, default=500000)  
parser.add_argument('--eval_episodes', type=int, default=25) 
parser.add_argument('--eval_every', type=int, default=5000)
parser.add_argument('--finetune_episode_steps', type=int, default=1000)
parser.add_argument('--log_path', type=str, default='./experiments/')


# Temporal parameters
parser.add_argument('--ensemble_size', type=int, default=5) 

args = parser.parse_args()
args.seed = np.random.randint(1e3) if args.seed == -1 else args.seed

if args.enable_wandb:
    wandb.init(project=args.project, config=args, group=args.group, name='{}_{}_seed{}'.format(args.agent, args.env_name, args.seed))
experiments_dir = args.log_path + args.project + args.group + '/' + '{}_{}_seed{}'.format(args.agent, args.env_name.split('-')[0], args.seed) + '/'
logger.configure(experiments_dir)
logger.log('This running starts with parameters:')
logger.log('----------------------------------------')
for k, v in args._get_kwargs():
    logger.log('- ' + str(k) + ': ' + str(v))
logger.log('----------------------------------------')
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

def eval_agent(mode='plan'):
    returns = []
    for i in range(args.eval_episodes):
        agent.reset()
        obs = env.reset()[0]
        for step in range(50):
            if mode == 'plan':
                action = agent.plan(obs['observation'], obs['desired_goal'])
            else:
                action = agent.get_action(obs['observation'], obs['desired_goal'])
            obs, reward, _, _, info = env.step(action)
            returns.append(reward.item())
    mean_return = np.array(returns).sum() / args.eval_episodes
    return {'return (' + mode + ')': mean_return}

t_start = time.time()
epoch = 0
total_step = 0
if args.pretrain:
    logger.log('Pretraining ...')
    for i in tqdm(range(0, args.pretrain_steps), mininterval=1):
        policy_eval_info, plan_eval_info = {}, {}
        training_info = agent.train_models()
        total_step += 1
        
        if (i + 1) % args.eval_every == 0:
            policy_eval_info = eval_agent('policy')
            # plan_eval_info = eval_agent('plan')
            logger.log('The performance after pretraining ' + str(i+1) + ' steps: ', policy_eval_info, plan_eval_info)
        
            # Logs
            log_info = {**training_info, **policy_eval_info, **plan_eval_info}
            epoch += 1
            logger.record_tabular('total steps', total_step)
            logger.record_tabular('training epoch', epoch)
            logger.record_tabular('epoch time (min)', (time.time() - t_start)/60)
            for key, val in log_info.items():
                if type(val) == torch.Tensor:
                    logger.record_tabular(key, val.mean().item())
                else:
                    logger.record_tabular(key, val)
            logger.dump_tabular()
            if args.enable_wandb:
                wandb.log(log_info)
            t_start = time.time()
    agent.save(args.log_path + args.project + '/' + env_info['env_name'] + '-pretrain')
else:
    agent.load(args.log_path + args.project + '/' + env_info['env_name'] + '-pretrain')
    #logger.log('Before finetuning, we check the dyanmics and the CGAN are well trained.')
    #policy_eval_info = eval_agent('policy')
    #plan_eval_info = eval_agent('plan')
    #logger.log('The performance before finetuning: ', policy_eval_info, plan_eval_info)

agent.dynamics.num_models = args.ensemble_size

if hasattr(agent, 'produce_intra_traj') and hasattr(agent, 'produce_inter_traj'):
    logger.log('Finetuning ...')
    t_start = time.time()
    epoch = 0
    for i in range(40): 
        # Reanalysis
        intra_traj_info, inter_traj_info = {}, {}
        intra_traj_info = agent.produce_intra_traj(num_traj=1000) 
        inter_traj_info = agent.produce_inter_traj(num_traj=1000)
        
        # Finetune
        for _ in tqdm(range(args.finetune_episode_steps)):
            ft_training_info = agent.finetune_models()
            total_step += 1

        # Evaluation
        ft_eval_info = eval_agent('policy')
        logger.log('The performance after fine-tuning ' + str((i+1) * args.finetune_episode_steps) + ' steps: ', ft_eval_info)
        epoch += 1
        logger.record_tabular('total steps', total_step)
        logger.record_tabular('finetuning epoch', epoch)
        logger.record_tabular('epoch time (min)', (time.time() - t_start)/60)
        log_info = {**ft_training_info, **ft_eval_info}
        for key, val in log_info.items():
            if type(val) == torch.Tensor:
                logger.record_tabular(key, val.mean().item())
            else:
                logger.record_tabular(key, val)
        logger.dump_tabular()
        
        if args.enable_wandb:
            wandb.log({**intra_traj_info, **inter_traj_info, 
                       **ft_eval_info, **ft_training_info})
        t_start = time.time()


