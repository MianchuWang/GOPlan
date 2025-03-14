import numpy as np
import gym

gym_robotics = ['FetchReach-v1', 'FetchPush-v1', 'FetchPickAndPlace-v1',
                'FetchSlide-v1', 'HandReach-v0', 
                'FetchPushOOD-Right2Right-v1', 'FetchPushOOD-Left2Left-v1',
                'FetchPushOOD-Left2Right-v1', 'FetchPushOOD-Right2Left-v1',
                'FetchPickOOD-Right2Right-v1', 'FetchPickOOD-Right2Left-v1',
                'FetchPickOOD-Left2Left-v1', 'FetchPickOOD-Left2Right-v1']
gym_stack = ['FetchStack2']
points_envs = ['PointRooms', 'PointReach', 'PointCross']
sawyer_envs = ['SawyerReach', 'SawyerDoor']
dmc_envs = ['Reacher-v2']

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
    elif env_name.startswith('PointCross'):
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
    else:
        raise Exception('Invalid environment.')
    



def return_gym_robotics_env(env_name, render_mode):
    # import gymnasium as gym  
    class GymWrapper(gym.RewardWrapper):
        def reward(self, reward):
            # return = 1 if success else 0
            return reward + 1
    if 'OOD' in env_name:
        kwargs = get_ood_config(env_name)
        if 'Push' in env_name:
            from wgcsl.envs.fetch_ood import FetchPushOODEnv
            env = FetchPushOODEnv(**kwargs)
        else:
            from wgcsl.envs.fetch_ood import FetchPickOODEnv
            env = FetchPickOODEnv(**kwargs)
    else:
        env = gym.make(env_name)

    return GymWrapper(env), \
           {'env_name': env_name,
            'state_dim': env.observation_space['observation'].shape[0],
            'goal_dim': env.observation_space['desired_goal'].shape[0],
            'ac_dim': env.action_space.shape[0],
            'max_steps': env._max_episode_steps,
            'get_goal_from_state': get_goal_from_state(env_name),
            'compute_reward': lambda x, y, z : env.compute_reward(x, y, None) + 1}
    
def return_gym_stack_env(env_name, render_mode):
    # import gymnasium as gym
    import envs.gym_fetch_stack.envs as fetch_stack
    if env_name == 'FetchStack2':
        env = fetch_stack.FetchStack2Env('sparse')
        return env, {'env_name': env_name,
                     'state_dim': env.observation_space['observation'].shape[0],
                     'goal_dim': env.observation_space['desired_goal'].shape[0],
                     'ac_dim': env.action_space.shape[0],
                     'max_steps': 50,
                     'get_goal_from_state': lambda x : np.concatenate([x[..., 10:13], x[..., 25:28]], axis=-1),
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
    elif env_name.startswith('PointCross'):
        from envs.wgcsl_envs.point2d import Point2DWallEnv
        kwargs={
            'action_scale': 1,
            'wall_shape': 'cross', 
            'wall_thickness': 1,
            'target_radius':0.5,
            'ball_radius':0.3,
            'boundary_dist':10,
            'render_size': 512,
            'wall_color': 'darkgray',
            'bg_color': 'white',
            'images_are_rgb': True,
            'render_onscreen': False,
            'show_goal': True,
            'get_image_base_render_size': (48, 48),
            'fixed_goal': (-10,0),
            'fixed_goal_set': None,
            'fixed_init_position': (0,10), 
            'randomize_position_on_reset': False
        }
        env = PointGoalWrapper(Point2DWallEnv(**kwargs))
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


def get_ood_config(env_name):
    if env_name == 'FetchPushOOD-Right2Left-v1':
        kwargs={
            'goal_type': 'left',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    elif env_name == 'FetchPushOOD-Left2Right-v1':
        kwargs={
            'goal_type': 'right',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    elif env_name == 'FetchPushOOD-Left2Left-v1':
        kwargs={
            'goal_type': 'left',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    elif env_name == 'FetchPushOOD-Right2Right-v1':
        kwargs={
            'goal_type': 'right',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    elif env_name == 'FetchPickOOD-Right2Right-v1':
        kwargs={
            'goal_type': 'right',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    elif env_name == 'FetchPickOOD-Right2Left-v1':
        kwargs={
            'goal_type': 'left',
            'initial_type': 'right',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    elif env_name == 'FetchPickOOD-Left2Left-v1':
        kwargs={
            'goal_type': 'left',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    elif env_name == 'FetchPickOOD-Left2Right-v1':
        kwargs={
            'goal_type': 'right',
            'initial_type': 'left',
            'obj_range': 0.15,
            'target_range': 0.15 
         }
    else:
        raise NotImplementedError

    return kwargs