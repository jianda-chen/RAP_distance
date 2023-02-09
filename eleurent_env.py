import gym
from gym import core, spaces
import highway_env
from matplotlib import pyplot as plt
import time
import numpy as np



def _spec_to_box(spec):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0)
    high = np.concatenate(maxs, axis=0)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=np.float32)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class EleurentHighWay(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        from_pixels=True,
        height=84,
        width=84,
        frame_skip=3,
        max_episode_steps = 1000,
    ):
        assert domain_name == 'eleurent' and task_name == "highway"
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._frame_skip = frame_skip
        self._max_episode_steps = max_episode_steps  # DMC uses this


        config = highway_env.envs.highway_env.HighwayEnv.default_config()
        config["offscreen_rendering"] = True
        config["lanes_count"] = 4
        # config["simulation_frequency"] = 1
        config["policy_frequency"] = 15.
        # config["duration"] = 1000
        config["screen_width"] = 84 
        config["screen_height"] = 84
        config["scaling"] = 5.5 / (150 / 84)
        config["action"]= {
                "type": "ContinuousAction",
            }
        

        self._env = highway_env.envs.highway_env.HighwayEnv(config)

        # true and normalized action spaces
        self._true_action_space = spaces.Box(self._env.action_space.low, self._env.action_space.high, dtype=np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=255, shape=[3, height, width], dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values()
            )

        # self._internal_state_space = spaces.Box(
        #     low=-np.inf,
        #     high=np.inf,
        #     shape=self._env.physics.get_state().shape,
        #     dtype=np.float32
        # )

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self,):
        if self._from_pixels:
            obs = self._env.render(mode="rgb_array")
            obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def internal_state_space(self):
        return self._internal_state_space

    @property
    def action_space(self):
        return self._norm_action_space

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        all_reward = 0.
        # extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            _, reward, done, info = self._env.step(action)
            all_reward += reward or 0
            if done:
                break
        obs = self._get_obs()
        return obs, reward, done, info

    def reset(self):
        time_step = self._env.reset()
        obs = self._get_obs()
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
