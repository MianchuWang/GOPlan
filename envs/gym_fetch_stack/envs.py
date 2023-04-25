import numpy as np
from envs.gym_fetch_stack import fetch_stack_env, rotations, robot_env
from envs.gym_fetch_stack import utils as envs_utils
from gymnasium import utils

DISTANCE_THRESHOLD = 0.04

class FetchStack1TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack1.xml', num_blocks=1, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack2TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack3TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack4TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack5TrainerOneThirdIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.33)
        utils.EzPickle.__init__(self)


class FetchStack5TrainerOneTenthIsStackingEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.1)
        utils.EzPickle.__init__(self)


class FetchStack2TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack3TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack4TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack5TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack6TrainerEasyEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],
            'object5:joint': [1.25, 0.53, 0.70, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack6.xml', num_blocks=6, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, goals_on_stack_probability=0.0)
        utils.EzPickle.__init__(self)


class FetchStack1Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack1.xml', num_blocks=1, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchStack2Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
    
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = envs_utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        obs = np.concatenate([
            grip_pos,
            gripper_state,
            grip_velp,
            gripper_vel,
        ])

        achieved_goal = []

        for i in range(self.num_blocks):

            object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
            # rotations
            object_i_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.object_names[i]))
            # velocities
            object_i_velp = self.sim.data.get_site_xvelp(self.object_names[i]) * dt
            object_i_velr = self.sim.data.get_site_xvelr(self.object_names[i]) * dt
            # gripper state
            object_i_rel_pos = object_i_pos - grip_pos
            object_i_velp -= grip_velp

            obs = np.concatenate([
                obs,
                object_i_pos.ravel(),
                object_i_rel_pos.ravel(),
                object_i_rot.ravel(),
                object_i_velp.ravel(),
                object_i_velr.ravel()
            ])

            achieved_goal = np.concatenate([
                achieved_goal, object_i_pos.copy()
            ])

        achieved_goal = np.concatenate([achieved_goal, grip_pos.copy()])

        achieved_goal = np.squeeze(achieved_goal)

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal[:-3].copy(),
            'desired_goal': self.goal[:-3].copy(),
        }
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False

        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal[:-3]),
        }
        # info = {}
        reward = self.compute_reward(obs['achieved_goal'], self.goal[:-3], info)
        return obs, reward, done, False, info
    
    def reset(self, seed=None):
        self.goal, goals, number_of_goals_along_stack = self._sample_goal(return_extra_info=True)

        if number_of_goals_along_stack == 0 or not self.allow_blocks_on_stack:
            number_of_blocks_along_stack = 0
        elif number_of_goals_along_stack < self.num_blocks:
            number_of_blocks_along_stack = np.random.randint(0, number_of_goals_along_stack+1)
        else:
            number_of_blocks_along_stack = np.random.randint(0, number_of_goals_along_stack)

        self.sim.set_state(self.initial_state)

        prev_x_positions = [goals[0][:2]]
        for i, obj_name in enumerate(self.object_names):
            object_qpos = self.sim.data.get_joint_qpos('{}:joint'.format(obj_name))
            assert object_qpos.shape == (7,)
            object_qpos[2] = 0.425

            if i < number_of_blocks_along_stack:
                object_qpos[:3] = goals[i]
                object_qpos[:2] += np.random.normal(loc=0, scale=0.002, size=2)
            else:
                object_xpos = self.initial_gripper_xpos[:2].copy()

                while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1 \
                        or np.any([np.linalg.norm(object_xpos - other_xpos) < 0.05 for other_xpos in prev_x_positions]):
                    object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                         size=2)
                object_qpos[:2] = object_xpos

            prev_x_positions.append(object_qpos[:2])
            self.sim.data.set_joint_qpos('{}:joint'.format(obj_name), object_qpos)

        self.sim.forward()

        obs = self._get_obs()
        return obs, None
    
    def sub_goal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        #goal_a = goal_a[..., :-3]
        #goal_b = goal_b[..., :-3]
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3:(i + 1) * 3].shape == goal_a[..., (i+1) * 3:(i + 2) * 3].shape

        return [
            np.linalg.norm(goal_a[..., i*3:(i+1)*3] - goal_b[..., i*3:(i+1)*3], axis=-1) for i in range(self.num_blocks)
        ]
    
    def compute_reward(self, achieved_goal, goal, info):
        distances = self.sub_goal_distances(achieved_goal, goal)
        reward = np.min([-(d > self.distance_threshold).astype(np.float32) for d in distances], axis=0)
        reward = np.asarray(reward)
        return reward + 1

    
    def _sample_goal(self, return_extra_info=False):

        max_goals_along_stack = self.num_blocks
        #TODO was 2
        if self.all_goals_always_on_stack:
            min_goals_along_stack = self.num_blocks
        else:
            min_goals_along_stack = 2


        # TODO: was 0.66
        if np.random.uniform() < 1.0 - self.goals_on_stack_probability:
            max_goals_along_stack = 1
            min_goals_along_stack = 0
        number_of_goals_along_stack = np.random.randint(min_goals_along_stack, max_goals_along_stack + 1)

        goal0 = None
        first_goal_is_valid = False
        while not first_goal_is_valid:
            goal0 = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            if self.num_blocks > 4:
                if np.linalg.norm(goal0[:2] - self.initial_gripper_xpos[:2]) < 0.09:
                    continue
            first_goal_is_valid = True

        # goal0[0] = goal0[0] - 0.05
        goal0 += self.target_offset
        goal0[2] = self.height_offset

        goals = [goal0]

        prev_x_positions = [goal0[:2]]
        goal_in_air_used = False
        for i in range(self.num_blocks - 1):
            if i < number_of_goals_along_stack - 1:
                goal_i = goal0.copy()
                goal_i[2] = self.height_offset + (0.05 * (i + 1))
            else:
                goal_i_set = False
                goal_i = None
                while not goal_i_set or np.any([np.linalg.norm(goal_i[:2] - other_xpos) < 0.06 for other_xpos in prev_x_positions]):
                    goal_i = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
                    goal_i_set = True

                goal_i += self.target_offset
                goal_i[2] = self.height_offset

                # TODO was 0.2
                if np.random.uniform() < 0.2 and not goal_in_air_used:
                    goal_i[2] += self.np_random.uniform(0.03, 0.1)
                    goal_in_air_used = True

            prev_x_positions.append(goal_i[:2])
            goals.append(goal_i)
        goals.append([0.0, 0.0, 0.0])
        if not return_extra_info:
            return np.concatenate(goals, axis=0).copy()
        else:
            return np.concatenate(goals, axis=0).copy(), goals, number_of_goals_along_stack

class FetchStack3Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchStack4Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchStack5Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchStack6Env(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],
            'object5:joint': [1.25, 0.53, 0.70, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack6.xml', num_blocks=6, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)


class FetchStack2TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack2.xml', num_blocks=2, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack3TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack3.xml', num_blocks=3, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack4TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack4.xml', num_blocks=4, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack5TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack5.xml', num_blocks=5, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)


class FetchStack6TestEnv(fetch_stack_env.FetchStackEnv, utils.EzPickle):
    def __init__(self, reward_type='incremental'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
            'object2:joint': [1.25, 0.53, 0.52, 1., 0., 0., 0.],
            'object3:joint': [1.25, 0.53, 0.58, 1., 0., 0., 0.],
            'object4:joint': [1.25, 0.53, 0.64, 1., 0., 0., 0.],
            'object5:joint': [1.25, 0.53, 0.70, 1., 0., 0., 0.],

        }
        fetch_stack_env.FetchStackEnv.__init__(
            self, 'fetch/stack6.xml', num_blocks=6, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.12, distance_threshold=DISTANCE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, all_goals_always_on_stack=True, allow_blocks_on_stack=False)
        utils.EzPickle.__init__(self)