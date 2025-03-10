from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import mujoco_py
from gym import Wrapper

from envs.wgcsl_envs.serializable import Serializable
from envs.wgcsl_envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)

from envs.wgcsl_envs.mujoco_env import MujocoEnv
import copy

from envs.wgcsl_envs.multitask_env import MultitaskEnv
import os.path as osp
from envs.wgcsl_envs import sawyer_door_hook

door_configs = {
    'all': dict(
            goal_low=(0,),
            goal_high=(.83,),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=False,
        )
}

class SawyerViews:
    @staticmethod
    def configure_viewer(cam, cam_pos):
        for i in range(3):
            cam.lookat[i] = cam_pos[i]
        cam.distance = cam_pos[3]
        cam.elevation = cam_pos[4]
        cam.azimuth = cam_pos[5]
        cam.trackbodyid = -1
    
    @staticmethod
    def robot_view(cam):
        rotation_angle = 90
        cam_dist = 1
        cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def third_person_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def top_down_view(cam):
        cam_dist = 0.2
        rotation_angle = 0
        cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def default_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 0.85, 0.30, cam_dist, -55, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)

class SawyerDoorGoalEnv(Wrapper):
    def __init__(self, fixed_start=True, fixed_goal=False, threshold=0.06):
        config_key = 'all'
        if fixed_start:
            if fixed_goal:
                config_key = 'fixed_start_fixed_goal'
            else:
                config_key = 'all'#'fixed_start'
        self.env = sawyer_door_hook.SawyerDoorHookEnv(**door_configs[config_key])
        Wrapper.__init__(self, self.env)
        self.threshold = threshold
        
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        reward = self.compute_rewards(action, ob)
        return ob, reward, done, info
    
    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        distance = self.goal_distance(achieved_goals, desired_goals)
        return -(distance > self.threshold).astype(float)
    
    def compute_reward(self, actions, obs):
        return self.compute_rewards(actions, obs)
  
    
    def goal_distance(self, states, goal_states):
        diff = states - goal_states
        return np.linalg.norm(diff[..., -1:], axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        puck_distances = np.array([self.door_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])

        statistics = OrderedDict()
        for stat_name, stat in [
            ('final door distance', puck_distances[:,-1]),
            ('min door distance', np.min(puck_distances, axis=-1)),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics
        

def main():
    e = SawyerDoorGoalEnv(discrete_action=True, fixed_start=True)
    for traj in range(20):
        desired_goal_state = e.sample_goal()
        states = []
        s = e.reset()
        for step in range(1):
            states.append(s)
            s, _, _, _ = e.step(e.action_space.sample())
            #e.render()
        states = np.stack(states)

if __name__ == "__main__":
    main()