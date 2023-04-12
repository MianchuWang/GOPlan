import numpy as np

gym_robotics = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1',
                'FetchSlide-v1', 'HandReach-v0']
gym_stack = ['FetchStack2Env']
points_envs = ['PointRooms', 'PointReach']
sawyer_envs = ['SawyerReach', 'SawyerDoor']
dmc_envs = ['Reacher-v2']
d4rl_antmaze_envs = ['antmaze-umaze-v2', 'antmaze-umaze-diverse-v2',
                     'antmaze-medium-play-v2', 'antmaze-medium-diverse-v2',
                     'antmaze-large-play-v2', 'antmaze-large-diverse-v2']

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
    elif env_name.startswith('SawyerReach'):
        return lambda x : x
    elif env_name.startswith('SawyerDoor'):
        return lambda x : x[..., -1:]
    elif env_name.startswith('Reacher'):
        return lambda x : x[..., -3:-1]
    elif env_name.startswith('PointRooms') or env_name.startswith('PointReach'):
        return lambda x : x
    else:
        raise Exception('Invalid environment. The environments options are', gym_robotics +
                        gym_stack + sawyer_envs + points_envs)

def return_environment(env_name, render_mode):
    if env_name in gym_robotics:
        return return_gym_robotics_env(env_name, render_mode)
    elif env_name in gym_stack:
        return return_gym_stack_env(env_name, render_mode)
    elif env_name in (sawyer_envs + points_envs + dmc_envs):
        return return_wgcsl_env(env_name, render_mode)
    elif env_name in d4rl_antmaze_envs:
        return return_d4rl_env(env_name, render_mode)
    else:
        raise Exception('Invalid environment.')


def return_gym_robotics_env(env_name, render_mode):
    import gymnasium as gym
    
    class GymWrapper(gym.RewardWrapper):
        def reward(self, reward):
            # return = 1 if success else 0
            return reward + 1
    env = gym.make(env_name, render_mode=render_mode)
    return GymWrapper(env), \
           {'env_name': env_name,
            'state_dim': env.observation_space['observation'].shape[0],
            'goal_dim': env.observation_space['desired_goal'].shape[0],
            'ac_dim': env.action_space.shape[0],
            'max_steps': env._max_episode_steps,
            'get_goal_from_state': get_goal_from_state(env_name),
            'compute_reward': lambda x, y, z : env.compute_reward(x, y, None) + 1}
    
def return_gym_stack_env(env_name, render_mode):
    import gymnasium as gym
    import envs.gym_fetch_stack.envs as fetch_stack
    if env_name == 'FetchStack2Env':
        env = fetch_stack.FetchStack2Env()
        return env, {'env_name': env_name,
                     'state_dim': env.observation_space['observation'].shape[0],
                     'goal_dim': env.observation_space['desired_goal'].shape[0],
                     'ac_dim': env.action_space.shape[0],
                     'max_steps': 50,
                     'get_goal_from_state': lambda x : np.concatenate([x[10:13], x[25:28], x[:3]]),
                     'compute_reward': lambda x, y, z : env.compute_reward(x, y, None) + 1}

def return_wgcsl_env(env_name, render_mode):
    from envs.wgcsl_envs.multi_world_wrapper import PointGoalWrapper, SawyerGoalWrapper, ReacherGoalWrapper
    if env_name.startswith('SawyerReach'):
        from envs.wgcsl_envs.sawyer_reach import SawyerReachXYZEnv
        env = SawyerGoalWrapper(SawyerReachXYZEnv())
    elif env_name.startswith('SawyerDoor'):
        from envs.wgcsl_envs.sawyer_door import SawyerDoorGoalEnv
        env = SawyerGoalWrapper(SawyerDoorGoalEnv())
    elif env_name.startswith('PointRooms'):
        from envs.wgcsl_envs.point2d import Point2DWallEnv
        env = PointGoalWrapper(Point2DWallEnv())
    elif env_name.startswith('PointReach'):
        from envs.wgcsl_envs.point2d import Point2DEnv
        env = PointGoalWrapper(Point2DEnv())
    elif env_name.startswith('Reacher-v2'):
        import gym
        env = ReacherGoalWrapper(gym.make(env_name))
    return env, {'env_name': env_name,
                 'state_dim': env.observation_space['observation'].shape[0],
                 'goal_dim': env.observation_space['desired_goal'].shape[0],
                 'ac_dim': env.action_space.shape[0],
                 'max_steps': 50,
                 'get_goal_from_state': get_goal_from_state(env_name),
                 'compute_reward': env.compute_reward}

def return_d4rl_env(env_name, render_mode):
    import gym, d4rl
    class AntWrapper(gym.ObservationWrapper):
        ''' Wrapper for exposing the goals of the AntMaze environment. '''
        def reset(self, seed, **kwargs):
            """Resets the environment, returning a modified observation using :meth:`self.observation`."""
            obs = self.env.reset(**kwargs)
            return self.observation(obs)
        def observation(self, observation):
            return {'observation': observation, 
                    'achieved_goal': observation[:2],
                    'desired_goal': np.array(self.env.target_goal)}
        def compute_reward(self, achieved_goal, desired_goal, info):
            reward = (np.linalg.norm(achieved_goal - desired_goal, axis=-1) <= 0.5).astype(np.float32)
            return reward
        @property
        def max_episode_steps(self):
            return self.env._max_episode_steps
    env = gym.make(env_name)
    return AntWrapper(env), {'env_name': env_name,
                             'state_dim': env.observation_space.shape[0],
                             'goal_dim': 2,
                             'ac_dim': env.action_space.shape[0],
                             'max_steps': env._max_episode_steps, 
                             'get_goal_from_state': lambda x : x[..., :2],
                             'compute_reward': lambda x, y, z: (np.linalg.norm(x - y, axis=-1) <= 0.5).astype(np.float32)}