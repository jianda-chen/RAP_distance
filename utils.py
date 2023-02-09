import torch
import numpy as np
import torch.nn as nn
import torch.distributions
import gym
import os
from collections import deque
import random
from gym import spaces


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs

def gym_action_space_log_prob(action_space, sample):
    assert isinstance(action_space, spaces.Box)
    high = action_space.high if action_space.dtype.kind == 'f' \
                else action_space.high.astype('int64') + 1
    sample = np.empty(action_space.shape)

    # Masking arrays which classify the coordinates according to interval
    # type
    unbounded   = ~action_space.bounded_below & ~action_space.bounded_above
    upp_bounded = ~action_space.bounded_below &  action_space.bounded_above
    low_bounded =  action_space.bounded_below & ~action_space.bounded_above
    bounded     =  action_space.bounded_below &  action_space.bounded_above
    assert unbounded.sum() == 0
    assert upp_bounded.sum() == 0
    assert low_bounded.sum() == 0

    sample = torch.tensor(sample)


    # sample[unbounded] = action_space.np_random.normal(
    #         size=unbounded[unbounded].shape)
    # sample[low_bounded] = action_space.np_random.exponential(
    #     size=low_bounded[low_bounded].shape) + action_space.low[low_bounded]

    # sample[upp_bounded] = -action_space.np_random.exponential(
    #     size=upp_bounded[upp_bounded].shape) + action_space.high[upp_bounded]

    # sample[bounded] = action_space.np_random.uniform(low=action_space.low[bounded],
    #                                     high=high[bounded],
    #                                     size=bounded[bounded].shape)
    bounded_dist = torch.distributions.Uniform(
        low=torch.tensor(action_space.low[bounded]),
        high=torch.tensor(high[bounded]))
    bounded_log_prob = bounded_dist.log_prob(sample)
    sample_log_prob = bounded_log_prob.sum(dim=-1, keepdim=True)
    return sample_log_prob.cpu().data.numpy()


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, is_framestack=False):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.is_framestack = is_framestack

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        if self.is_framestack:
            self.obses = np.empty((capacity,) + (obs_shape[0] + 3,) + obs_shape[1:], dtype=obs_dtype)
            self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        else:
            self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
            self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.log_action_probs = np.empty((capacity, 1), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, log_pi, curr_reward, reward, next_obs, done):
        if self.is_framestack:
            np.copyto(self.obses[self.idx][:-3], obs)
            np.copyto(self.obses[self.idx][-3:], next_obs[-3:])
        else:
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.log_action_probs[self.idx], log_pi)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        if self.is_framestack:
            obses = torch.as_tensor(self.obses[idxs][:, :-3], device=self.device).float()
            next_obses = torch.as_tensor(
                self.obses[idxs][:, 3:], device=self.device
            ).float()
        else:
            obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
            next_obses = torch.as_tensor(
                self.next_obses[idxs], device=self.device
            ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        log_pi = torch.as_tensor(self.log_action_probs[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, log_pi, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, log_pi, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.log_action_probs[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            if not self.is_framestack:
                self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.log_action_probs[start:end] = payload[3]
            self.rewards[start:end] = payload[4]
            self.curr_rewards[start:end] = payload[5]
            self.not_dones[start:end] = payload[6]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
