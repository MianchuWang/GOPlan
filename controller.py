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
        self.enable_wandb = enable_wandb

    def train(self):
        for i in tqdm(range(0, self.pretrain_steps)):
            policy_eval_info = {}
            plan_eval_info = {}
            training_info = self.agent.train_models()
            if i % 10000 == 0:
                policy_eval_info = self.eval('policy')
                plan_eval_info = self.eval('plan')
            if self.enable_wandb:
                wandb.log({**training_info, **plan_eval_info, **policy_eval_info})

    def eval(self, mode='plan'):
        returns = []
        for i in tqdm(range(self.eval_episodes)):
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
        print('return (' + mode + ') is ' + str(mean_return))
        return {'return (' + mode + ')': mean_return}
