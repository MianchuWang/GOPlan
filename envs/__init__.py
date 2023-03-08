import gymnasium as gym
import envs.gym_fetch_stack.envs as fetch_stack
import numpy as np

gym_robotics = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1',
                'FetchSlide-v1', 'HandReach-v0']
gym_stack = ['FetchStack2Env']

def get_goal_from_state(env_name):
    if env_name.startswith('FetchReach'):
        return lambda x : x[..., :3]
    elif env_name.startswith('FetchPush'):
        return lambda x : x[..., 3:6]
    elif env_name.startswith('FetchPickAndPlace'):
        return lambda x : x[..., 3:6]
    elif env_name.startswith('FetchSlide'):
        return lambda x : x[..., 3:6]
    elif env_name.startswith('HandReach'):
        return lambda x : x[..., -15:]
    else:
        raise Exception('Invalid environment.')

def return_environment(env_name, render_mode):
    if env_name in gym_robotics:
        env = gym.make(env_name, render_mode=render_mode)
        return GymWrapper(env), \
               {'env_name': env_name,
                'state_dim': env.observation_space['observation'].shape[0],
                'goal_dim': env.observation_space['desired_goal'].shape[0],
                'ac_dim': env.action_space.shape[0],
                'max_steps': env._max_episode_steps,
                'get_goal_from_state': get_goal_from_state(env_name),
                'compute_reward': lambda x, y, z : env.compute_reward(x, y, None) + 1}
    elif env_name in gym_stack:
        if env_name == 'FetchStack2Env':
            env = fetch_stack.FetchStack2Env()
            return env, {'env_name': env_name,
                         'state_dim': env.observation_space['observation'].shape[0],
                         'goal_dim': env.observation_space['desired_goal'].shape[0],
                         'ac_dim': env.action_space.shape[0],
                         'max_steps': 50,
                         'get_goal_from_state': lambda x : np.concatenate([x[10:13], x[25:28], x[:3]]),
                         'compute_reward': lambda x, y, z : env.compute_reward(x, y, None) + 1}
        else:
            raise Exception('Invalid fetch stack environment.')
    else:
        raise Exception('Invalid environment.')


class GymWrapper(gym.RewardWrapper):
    def reward(self, reward):
        # return = 1 if success else 0
        return reward + 1
