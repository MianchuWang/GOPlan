from collections import OrderedDict
import logging

import numpy as np
from gym import spaces
from pygame import Color

from wgcsl.envs.multitask_env import MultitaskEnv
from wgcsl.envs.serializable import Serializable
from wgcsl.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
)
from wgcsl.envs.pygame_viewer import PygameViewer
from wgcsl.envs.walls import VerticalWall, HorizontalWall


class Point2DEnv(MultitaskEnv, Serializable):
    """
    A little 2D point whose life goal is to reach a target.
    """

    def __init__(
            self,
            render_dt_msec=0,
            action_l2norm_penalty=0,  # disabled for now
            render_onscreen=False,
            render_size=84,
            get_image_base_render_size=None,
            reward_type="dense",
            action_scale=1.0,
            target_radius=0.60,
            boundary_dist=4,
            ball_radius=0.50,
            walls=None,
            fixed_goal=None,
            fixed_goal_set=None,
            fixed_goal_set_id=0,
            fixed_init_position=None,
            circle_radius=0,
            randomize_position_on_reset=True,
            images_are_rgb=False,  # else black and white
            show_goal=True,
            pointmass_color="blue",
            bg_color="black",
            wall_color="white",
            **kwargs
    ):
        if walls is None:
            walls = []
        if walls is None:
            walls = []
        if fixed_goal is not None:
            fixed_goal = np.array(fixed_goal)
        if fixed_goal_set is not None:
            self.fixed_set = True
            #### circle
            # circle_radius = 20
            angle = np.linspace(0,1,10) * np.pi
            x,y = circle_radius * np.cos(angle), circle_radius * np.sin(angle)
            gs = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], 1)
            gs2 = np.concatenate([x.reshape(-1,1), - y.reshape(-1,1)], 1)
            goals = np.concatenate([gs, gs2])
            if fixed_goal_set_id == 0:
                goal_tmp = goals
            elif fixed_goal_set_id == 1:
                goal_tmp = goals[:10]
            elif fixed_goal_set_id == 2:
                goal_tmp = goals[10:]
            elif fixed_goal_set_id == 3:
                goal_tmp = goals[15:]
            else:
                goal_tmp = goals

            self.fixed_goal_set = goal_tmp
            # import pdb;pdb.set_trace()
            # [(4,-4), (-4,4), (4,4), (-4,-4)]
        else:
            self.fixed_set = False
        if fixed_init_position is not None:
            fixed_init_position = np.array(fixed_init_position)
        if len(kwargs) > 0:
            LOGGER = logging.getLogger(__name__)
            LOGGER.log(logging.WARNING, "WARNING, ignoring kwargs:", kwargs)
        self.quick_init(locals())
        self.render_dt_msec = render_dt_msec
        self.action_l2norm_penalty = action_l2norm_penalty
        self.render_onscreen = render_onscreen
        self.render_size = render_size
        self.reward_type = reward_type
        self.action_scale = action_scale
        self.target_radius = target_radius
        self.boundary_dist = boundary_dist
        self.ball_radius = ball_radius
        self.walls = walls
        self.fixed_goal = fixed_goal
        self._fixed_init_position = fixed_init_position
        self.randomize_position_on_reset = randomize_position_on_reset
        self.images_are_rgb = images_are_rgb
        self.show_goal = show_goal
        self.pointmass_color = pointmass_color
        self.bg_color = bg_color
        self._wall_color = wall_color
        self.render_drawer = None

        self.max_target_distance = self.boundary_dist - self.target_radius

        self._target_position = np.zeros(2)
        self._position = np.zeros(2)

        u = np.ones(2)
        self.action_space = spaces.Box(-u, u, dtype=np.float32)

        o = self.boundary_dist * np.ones(2)
        self.obs_range = spaces.Box(-o, o, dtype='float32')
        self.observation_space = spaces.Dict([
            ('observation', self.obs_range),
            ('desired_goal', self.obs_range),
            ('achieved_goal', self.obs_range),
            ('state_observation', self.obs_range),
            ('state_desired_goal', self.obs_range),
            ('state_achieved_goal', self.obs_range),
        ])

        if get_image_base_render_size:
            base_width, base_height = get_image_base_render_size
            self._drawer = PygameViewer(
                screen_width=base_width,
                screen_height=base_height,
                x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
            self._fixed_get_image_render_size = True
        else:
            self._drawer = None
            self._fixed_get_image_render_size = False

    def step(self, velocities):
        assert self.action_scale <= 1.0
        velocities = np.clip(velocities, a_min=-1, a_max=1) * self.action_scale
        new_position = self._position + velocities
        orig_new_pos = new_position.copy()
        for wall in self.walls:
            new_position = wall.handle_collision(
                self._position, new_position
            )
        if sum(new_position != orig_new_pos) > 1:
            """
            Hack: sometimes you get caught on two walls at a time. If you
            process the input in the other direction, you might only get
            caught on one wall instead.
            """
            new_position = orig_new_pos.copy()
            for wall in self.walls[::-1]:
                new_position = wall.handle_collision(
                    self._position, new_position
                )

        self._position = new_position
        self._position = np.clip(
            self._position,
            a_min=-self.boundary_dist,
            a_max=self.boundary_dist,
        )
        distance_to_target = np.linalg.norm(
            self._position - self._target_position
        )
        is_success = distance_to_target < self.target_radius

        ob = self._get_obs()
        reward = self.compute_reward(velocities, ob)
        info = {
            'radius': self.target_radius,
            'target_position': self._target_position,
            'distance_to_target': distance_to_target,
            'velocity': velocities,
            'speed': np.linalg.norm(velocities),
            'is_success': is_success,
        }
        done = False
        return ob, reward, done, info


    def reset(self):
        self._target_position = self.sample_goal()['state_desired_goal']
        if self.randomize_position_on_reset:
            self._position = self._sample_position(
                self.obs_range.low,
                self.obs_range.high,
            )
        else:
            self._position = self._fixed_init_position

        return self._get_obs()

    def _position_inside_wall(self, pos):
        for wall in self.walls:
            if wall.contains_point(pos):
                return True
        return False

    def _sample_position(self, low, high):
        pos = np.random.uniform(low, high)
        while self._position_inside_wall(pos) is True:
            pos = np.random.uniform(low, high)
        return pos

    def _get_obs(self):
        return dict(
            observation=self._position.copy(),
            desired_goal=self._target_position.copy(),
            achieved_goal=self._position.copy(),
            state_observation=self._position.copy(),
            state_desired_goal=self._target_position.copy(), 
            state_achieved_goal=self._position.copy(),
        )

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        d = np.linalg.norm(achieved_goals - desired_goals, axis=-1)
        if self.reward_type == "sparse":
            return -(d > self.target_radius).astype(np.float32)
        elif self.reward_type == "dense":
            return -d
        elif self.reward_type == 'vectorized_dense':
            return -np.abs(achieved_goals - desired_goals)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'radius',
            'target_position',
            'distance_to_target',
            'velocity',
            'speed',
            'is_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    def get_contextual_diagnostics(self, paths, contexts):
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"
        values = []
        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key]
            goal = contexts[i][goal_key]
            distance = np.linalg.norm(state - goal)
            values.append(distance)
        diagnostics_key = goal_key + "/final/distance"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))

        values = []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key]
                goal = contexts[i][goal_key]
                distance = np.linalg.norm(state - goal)
                values.append(distance)
        diagnostics_key = goal_key + "/distance"
        diagnostics.update(create_stats_ordered_dict(
            diagnostics_key,
            values,
        ))
        return diagnostics

    def goal_conditioned_diagnostics(self, paths, goals):
        statistics = OrderedDict()
        distance_to_target_list = []
        is_success_list = []
        for path, goal in zip(paths, goals):
            distance_to_target = np.linalg.norm(
                path['observations'] - goal,
                axis=1
            )
            is_success = distance_to_target < self.target_radius
            distance_to_target_list.append(distance_to_target)
            is_success_list.append(is_success)
        for stat_name, stat_list in [
            ('distance_to_target', distance_to_target_list),
            ('is_success', is_success_list),
        ]:
            statistics.update(create_stats_ordered_dict(
                stat_name,
                stat_list,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}/final'.format(stat_name),
                [s[-1:] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
            statistics.update(create_stats_ordered_dict(
                '{}/initial'.format(stat_name),
                [s[:1] for s in stat_list],
                always_show_all_stats=True,
                exclude_max_min=True,
            ))
        return statistics

    def get_goal(self):
        return {
            'desired_goal': self._target_position.copy(),
            'state_desired_goal': self._target_position.copy(),
        }

    def sample_goals(self, batch_size):
        if self.fixed_set:
            idxes = np.random.randint(len(self.fixed_goal_set), size=batch_size)
            goals = np.array(self.fixed_goal_set)[idxes]
        elif not self.fixed_goal is None:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.zeros((batch_size, self.obs_range.low.size))
            if len(self.walls) > 0:
                if batch_size > 1:
                    logging.warning("This is very slow!")
                for b in range(batch_size):
                    goals[b, :] = self._sample_position(
                        self.obs_range.low,
                        self.obs_range.high,
                    )
            else:
                goals = np.random.uniform(
                    self.obs_range.low,
                    self.obs_range.high,
                    size=(batch_size, self.obs_range.low.size),
                )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_position(self, pos):
        self._position[0] = pos[0]
        self._position[1] = pos[1]

    """Functions for ImageEnv wrapper"""

    def get_image(self, width=None, height=None):
        """Returns a black and white image"""
        if self._drawer is None or (
                not self._fixed_get_image_render_size
                and (self._drawer.width != width or self._drawer.height != height)
        ):
            if width != height:
                raise NotImplementedError()
            self._drawer = PygameViewer(
                screen_width=width,
                screen_height=height,
                x_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                y_bounds=(-self.boundary_dist - self.ball_radius, self.boundary_dist + self.ball_radius),
                render_onscreen=self.render_onscreen,
            )
        self.draw(self._drawer)
        if width and height:
            wh_size = (width, height)
        else:
            wh_size = None
        img = self._drawer.get_image(wh_size)
        if self.images_are_rgb:
            return img.transpose((1, 0, 2))
        else:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = (-r + b).transpose().flatten()
            return img

    def set_to_goal(self, goal_dict):
        goal = goal_dict["state_desired_goal"]
        self._position = goal
        self._target_position = goal
    
    def set_goal(self, goal_id):
        if self.fixed_set:
            if type(goal_id) == np.ndarray or type(goal_id) == list: 
                self._target_position = goal_id[0]
            else:
                self._target_position = self.fixed_goal_set[goal_id]

    def get_env_state(self):
        return self._get_obs()

    def set_env_state(self, state):
        position = state["state_observation"]
        goal = state["state_desired_goal"]
        self._position = position
        self._target_position = goal

    def draw(self, drawer):
        drawer.fill(Color(self.bg_color))
        if self.show_goal:
            drawer.draw_solid_circle(
                self._target_position,
                self.target_radius,
                Color('green'),
            )
        drawer.draw_solid_circle(
            self._position,
            self.ball_radius,
            Color(self.pointmass_color),
        )
        for wall in self.walls:
            drawer.draw_rect(
                wall.endpoint4,
                wall.endpoint1[0] - wall.endpoint4[0],
                - wall.endpoint1[1] + wall.endpoint2[1],
                Color(self._wall_color),
                thickness=1,
            )
        drawer.render()

    def render(self, mode='human', close=False):
        if close:
            self.render_drawer = None
            return

        if self.render_drawer is None or self.render_drawer.terminated:
            self.render_drawer = PygameViewer(
                self.render_size,
                self.render_size,
                x_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                y_bounds=(-self.boundary_dist-self.ball_radius, self.boundary_dist+self.ball_radius),
                render_onscreen=True,
            )
        self.draw(self.render_drawer)
        self.render_drawer.tick(self.render_dt_msec)
        if mode != 'interactive':
            self.render_drawer.check_for_exit()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'distance_to_target',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    """Static visualization/utility methods"""

    @staticmethod
    def true_model(state, action):
        velocities = np.clip(action, a_min=-1, a_max=1)
        position = state
        new_position = position + velocities
        return np.clip(
            new_position,
            a_min=-Point2DEnv.boundary_dist,
            a_max=Point2DEnv.boundary_dist,
        )

    @staticmethod
    def true_states(state, actions):
        real_states = [state]
        for action in actions:
            next_state = Point2DEnv.true_model(state, action)
            real_states.append(next_state)
            state = next_state
        return real_states

    @staticmethod
    def plot_trajectory(ax, states, actions, goal=None):
        assert len(states) == len(actions) + 1
        x = states[:, 0]
        y = -states[:, 1]
        num_states = len(states)
        plasma_cm = plt.get_cmap('plasma')
        for i, state in enumerate(states):
            color = plasma_cm(float(i) / num_states)
            ax.plot(state[0], -state[1],
                    marker='o', color=color, markersize=10,
                    )

        actions_x = actions[:, 0]
        actions_y = -actions[:, 1]

        ax.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1],
                  scale_units='xy', angles='xy', scale=1, width=0.005)
        ax.quiver(x[:-1], y[:-1], actions_x, actions_y, scale_units='xy',
                  angles='xy', scale=1, color='r',
                  width=0.0035, )
        ax.plot(
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                Point2DEnv.boundary_dist,
            ],
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )
        ax.plot(
            [
                Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            [
                -Point2DEnv.boundary_dist,
                -Point2DEnv.boundary_dist,
            ],
            color='k', linestyle='-',
        )

        if goal is not None:
            ax.plot(goal[0], -goal[1], marker='*', color='g', markersize=15)
        ax.set_ylim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )
        ax.set_xlim(
            -Point2DEnv.boundary_dist - 1,
            Point2DEnv.boundary_dist + 1,
        )

    def initialize_camera(self, init_fctn):
        pass


class Point2DWallEnv(Point2DEnv):
    """Point2D with walls"""

    def __init__(
            self,
            wall_shape="",
            wall_thickness=1.0,
            inner_wall_max_dist=1,
            **kwargs
    ):
        self.quick_init(locals())
        super().__init__(**kwargs)
        self.inner_wall_max_dist = inner_wall_max_dist
        self.wall_shape = wall_shape
        self.wall_thickness = wall_thickness
        if wall_shape == "u":
            self.walls = [
                # Right wall
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "-" or wall_shape == "h":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "--":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist,
                    self.inner_wall_max_dist,
                )
            ]
        if wall_shape == "---":
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "big-u":
            self.walls = [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "easy-u":
            self.walls = [
                VerticalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Left wall
                VerticalWall(
                    self.ball_radius,
                    -self.inner_wall_max_dist*2,
                    -self.inner_wall_max_dist*0.5,
                    self.inner_wall_max_dist,
                    self.wall_thickness
                ),
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "big-h":
            self.walls = [
                # Bottom wall
                HorizontalWall(
                    self.ball_radius,
                    self.inner_wall_max_dist,
                    -self.inner_wall_max_dist*2,
                    self.inner_wall_max_dist*2,
                ),
            ]
        if wall_shape == "box":
            self.walls = [
                # Bottom wall
                VerticalWall(
                    self.ball_radius,
                    0,
                    0,
                    0,
                    self.wall_thickness
                ),
            ]
        if wall_shape == 'four-room-v0':
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist,
                    -self.ball_radius * 6,
                    self.wall_thickness
                ),
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.ball_radius * 1.5,
                    self.ball_radius * 1.5,
                    self.wall_thickness
                ),
                HorizontalWall(
                    self.ball_radius,
                    0,
                    self.ball_radius * 6,
                    self.boundary_dist,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    0,
                    self.ball_radius * 6,
                    self.boundary_dist,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    0,
                    -self.ball_radius * 1.5,
                    self.ball_radius * 1.5,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist,
                    -self.ball_radius * 6,
                    self.wall_thickness
                )

            ]
        if wall_shape == 'four-room-v1':
            self.walls = [
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist * 1.2,
                    -self.boundary_dist * 1.05,
                    self.wall_thickness
                ),
                HorizontalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist * 0.17,
                    self.boundary_dist * 0.17,
                    self.wall_thickness
                ),
                HorizontalWall(
                    self.ball_radius,
                    0,
                    self.boundary_dist * 1.05,
                    self.boundary_dist * 1.2,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    0,
                    self.boundary_dist * 1.05,
                    self.boundary_dist * 1.2,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist * 0.17,
                    self.boundary_dist * 0.17,
                    self.wall_thickness
                    
                ),
                VerticalWall(
                    self.ball_radius,
                    0,
                    -self.boundary_dist * 1.2,
                    -self.boundary_dist * 1.05,
                    self.wall_thickness
                )
            ]
            
        if wall_shape == 'cross':
            self.walls = [
                 VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist * 0.2,
                    -self.boundary_dist,
                    -self.boundary_dist * 0.2,
                    1
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist * 0.2,
                    -self.boundary_dist * 1.0,
                    -self.boundary_dist * 0.2,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    -self.boundary_dist * 0.2,
                    self.boundary_dist * 0.2,
                    self.boundary_dist * 1,
                    1
                ),
                HorizontalWall(
                    self.ball_radius,
                    self.boundary_dist * 0.2,
                    -self.boundary_dist * 1.0,
                    -self.boundary_dist * 0.2,
                    self.wall_thickness
                ),

                 VerticalWall(
                    self.ball_radius,
                    self.boundary_dist * 0.2,
                    -self.boundary_dist,
                    -self.boundary_dist * 0.2,
                    1
                ),
                HorizontalWall(
                    self.ball_radius,
                    -self.boundary_dist * 0.2,
                    self.boundary_dist * 0.2,
                    self.boundary_dist * 1.0,
                    self.wall_thickness
                ),
                VerticalWall(
                    self.ball_radius,
                    self.boundary_dist * 0.2,
                    self.boundary_dist * 0.2,
                    self.boundary_dist * 1,
                    1
                ),
                HorizontalWall(
                    self.ball_radius,
                    self.boundary_dist * 0.2,
                    self.boundary_dist * 0.2,
                    self.boundary_dist * 1,
                    self.wall_thickness
                ),
            ]
        if wall_shape == "none":
            self.walls = []


# if __name__ == "__main__":
#     import gym
#     import matplotlib.pyplot as plt

#     # e = gym.make('Point2D-Box-Wall-v1')
#     # e = gym.make('Point2D-Big-UWall-v1')
#     e = gym.make('Point2D-Easy-UWall-v1')
#     for i in range(1000):
#         e.reset()
#         for j in range(5):
#             e.step(np.random.rand(2))
#             e.render()
#             im = e.get_image()
