import gymnasium as gym

gym_robotics = ['FetchReach-v3', 'FetchPush-v2', 'FetchPickAndPlace-v2',
                'FetchSlide-v2', 'HandReach-v0']
bandit_env = ['bandit']

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
               {'state_dim': env.observation_space['observation'].shape[0],
                'goal_dim': env.observation_space['desired_goal'].shape[0],
                'ac_dim': env.action_space.shape[0],
                'max_steps': env._max_episode_steps,
                'get_goal_from_state': get_goal_from_state(env_name),
                'compute_reward': lambda x, y, z : env.compute_reward(x, y, None) + 1}
    elif env_name in bandit_env:
        from envs.bandit import Bandit
        env = Bandit()
        return env, \
               {'state_dim': env.state_dim,
                'goal_dim': env.goal_dim,
                'ac_dim': env.ac_dim,
                'max_steps': env._max_episode_steps,
                'get_goal_from_state': lambda x : x,
                'compute_reward': None}
    else:
        raise Exception('Invalid environment.')


class GymWrapper(gym.RewardWrapper):
    def reward(self, reward):
        # return = 1 if success else 0
        return reward + 1
