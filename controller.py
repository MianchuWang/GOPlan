import numpy as np
import wandb

from tqdm import tqdm

class Controller:
    def __init__(self, pretrain_steps, eval_episodes, env, env_info, agent, buffer, enable_wandb):
        self.pretrain_steps = pretrain_steps
        self.eval_episodes = eval_episodes
        self.env = env
        self.env_info = env_info
        self.agent = agent
        self.buffer = buffer
        self.finetune_episode_steps = 500
        self.enable_wandb = enable_wandb

    def train(self):
        
        print('Pretraining ...')
        for i in tqdm(range(0, self.pretrain_steps)):
            policy_eval_info = {}
            plan_eval_info = {}
            training_info = self.agent.pretrain_models()
            if i % 20000 == 0:
                policy_eval_info = self.eval('policy')
                #plan_eval_info = self.eval('plan')
                print('The performance after pretraining ' + str(i) + ' steps: ', 
                        {**policy_eval_info, **plan_eval_info})
            if self.enable_wandb:
                wandb.log({**training_info, **policy_eval_info, **plan_eval_info})
        self.agent.save(self.env_info['env_name'] + '-pretrain')
        
        self.agent.load(self.env_info['env_name'] + '-pretrain')
        policy_eval_info = self.eval('policy')
        plan_eval_info = self.eval('plan')
        print('The performance after pretraining is ', {**policy_eval_info, **plan_eval_info})
        
        print('Finetuning ...')
        for i in range(50):
            # Intra-reanalysis
            intra_traj_info, inter_traj_info = {}, {}
            intra_traj_info = self.agent.produce_intra_traj(num_traj=2000)
            inter_traj_info = self.agent.produce_inter_traj(num_traj=2000)
            if self.enable_wandb:
                wandb.log({**intra_traj_info, **inter_traj_info})
            
            # Finetune
            for _ in tqdm(range(self.finetune_episode_steps)):
                ft_training_info = self.agent.finetune_models()
                if self.enable_wandb:
                    wandb.log(ft_training_info)

            # Evaluation
            ft_eval_info = self.eval('policy')
            print('The performance after fine-tuning ' + str((i+1) * self.finetune_episode_steps) + ' steps: ', ft_eval_info)
            if self.enable_wandb:
                wandb.log({**ft_eval_info})

        self.agent.save(self.env_info['env_name'] + '-finetune')

    def eval(self, mode='plan'):
        returns = []
        for i in range(self.eval_episodes):
            self.agent.reset()
            obs = self.env.reset(seed=np.random.randint(1e10))[0]
            for step in range(self.env_info['max_steps']):
                if mode == 'plan':
                    action = self.agent.plan(obs['observation'], obs['desired_goal'])
                else:
                    action = self.agent.get_action(obs['observation'], obs['desired_goal'])
                obs, reward, _, _, info = self.env.step(action)
                returns.append(reward)

        mean_return = np.array(returns).sum() / self.eval_episodes
        return {'return (' + mode + ')': mean_return}
