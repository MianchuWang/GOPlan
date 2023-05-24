import numpy as np
import wandb
import logger
import torch
from tqdm import tqdm
import time

class Controller:
    def __init__(self, pretrain_steps, eval_episodes, eval_every, experiments_dir, env, env_info, agent, buffer, enable_wandb):
        self.pretrain_steps = pretrain_steps
        self.eval_episodes = eval_episodes
        self.eval_every = eval_every
        self.experiments_dir = experiments_dir
        self.env = env
        self.env_info = env_info
        self.agent = agent
        self.buffer = buffer
        self.finetune_episode_steps = 500
        self.enable_wandb = enable_wandb

    def train(self):
        logger.log('Pretraining ...')
        t_start = time.time()
        epoch = 0
        total_step = 0
        for i in tqdm(range(0, self.pretrain_steps)):
            policy_eval_info = {}
            plan_eval_info = {}
            training_info = self.agent.train_models()
            total_step += 1
            
            if (i + 1) % self.eval_every == 0:
                policy_eval_info = self.eval('policy')
                #plan_eval_info = self.eval('plan')
                logger.log('The performance after pretraining ' + str(i+1) + ' steps: ', policy_eval_info, plan_eval_info)
            
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

                if self.enable_wandb:
                    wandb.log(log_info)
                
                t_start = time.time()
        
        # self.agent.save(self.experiments_dir + self.env_info['env_name'] + '-pretrain')
        #self.agent.load(self.experiments_dir + self.env_info['env_name'] + '-pretrain')
        policy_eval_info = self.eval('policy')
        logger.log('The performance after pretraining is ', policy_eval_info)
        
        logger.log('Finetuning ...')
        if hasattr(self.agent, 'produce_intra_traj') and hasattr(self.agent, 'produce_inter_traj'):
            t_start = time.time()
            epoch = 0
            for i in range(20): 
                # Intra-reanalysis
                intra_traj_info, inter_traj_info = {}, {}
                intra_traj_info = self.agent.produce_intra_traj(num_traj=1000) 
                inter_traj_info = self.agent.produce_inter_traj(num_traj=1000) # 500
                
                # Finetune
                for _ in tqdm(range(self.finetune_episode_steps)):
                    ft_training_info = self.agent.finetune_models()
                    total_step += 1

                # Evaluation
                ft_eval_info = self.eval('policy')
                logger.log('The performance after fine-tuning ' + str((i+1) * self.finetune_episode_steps) + ' steps: ', ft_eval_info)
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
                
                if self.enable_wandb:
                    wandb.log({**intra_traj_info, **inter_traj_info, 
                               **ft_eval_info, **ft_training_info})
                t_start = time.time()


    def eval(self, mode='plan'):
        returns = []
        for i in range(self.eval_episodes):
            self.agent.reset()
            # obs = self.env.reset(seed=np.random.randint(1e10))[0]
            obs = self.env.reset()
            for step in range(50):
                if mode == 'plan':
                    action = self.agent.plan(obs['observation'], obs['desired_goal'])
                else:
                    action = self.agent.get_action(obs['observation'], obs['desired_goal'])
                if self.env_info['env_name'].startswith('antmaze'):
                    obs, reward, _, info = self.env.step(action)
                    returns.append(self.env.get_normalized_score(reward))
                    if reward == 1:
                        break
                else:
                    obs, reward, _, info = self.env.step(action)
                    returns.append(reward.item())
                #self.env.render()
        mean_return = np.array(returns).sum() / self.eval_episodes
        return {'return (' + mode + ')': mean_return}
    
    def eval_env(self, env, mode='plan', episodes=0):
        returns = []
        eval_episodes = max(self.eval_episodes, episodes)
        for i in range(eval_episodes):
            self.agent.reset()
            # obs = env.reset(seed=np.random.randint(1e10))[0]
            obs = env.reset()
            for step in range(50):
                if mode == 'plan':
                    action = self.agent.plan(obs['observation'], obs['desired_goal'])
                else:
                    action = self.agent.get_action(obs['observation'], obs['desired_goal'])
                if self.env_info['env_name'].startswith('antmaze'):
                    obs, reward, _, info = env.step(action)
                    returns.append(env.get_normalized_score(reward))
                    if reward == 1:
                        break
                else:
                    obs, reward, _, info = env.step(action)
                    returns.append(reward.item())
        mean_return = np.array(returns).sum() / eval_episodes
        return {'return (' + mode + ')': mean_return}
