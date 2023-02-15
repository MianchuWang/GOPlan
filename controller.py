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
        for i in tqdm(range(self.pretrain_steps)):
            training_info = self.agent.train_models()
            if i % 10000 == 0:
                eval_info = self.eval()
            if self.enable_wandb:
                wandb.log({**training_info, **eval_info})
                eval_info = {}

    def eval(self):
        returns = []
        for i in tqdm(range(self.eval_episodes)):
            self.agent.reset()
            obs = self.env.reset(seed=np.random.randint(1e10))[0]
            for step in range(self.env_info['max_steps']):
                action = self.agent.get_action(obs['observation'], obs['desired_goal'])
                obs, reward, _, _, info = self.env.step(action)
                returns.append(reward)

        mean_return = np.array(returns).sum() / self.eval_episodes
        print('The return is', mean_return)
        return {'return': mean_return}
