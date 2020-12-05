# This code was addapted from
# https://github.com/devendrachaplot/Neural-SLAM/blob/master/env/__init__.py
import torch
import numpy as np

from .habitat import construct_envs

#TODO
def make_vec_envs(args):
    envs = construct_envs(args)
    envs = VecPyTorch(envs, args.device)
    return envs
    

class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_spaces
        self.action_space = venv.action_space
        self.device = device

    def reset(self):
        r = self.venv.reset()
        obs = np.array([x[0] for x in r])
        info = [x[1] for x in r]
        # obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        actions = actions.cpu().numpy()
        # obs, reward, done, info = self.venv.step(actions)
        
        obses = np.array(self.venv.step(actions))
        shp = obses.shape
        obs, reward, done, info = obses.ravel(order='F').reshape(shp[-1], shp[0])
        # obs = np.array(obs).astype(np.float)
        # reward = np.array(reward).astype(np.float)
        # done = np.array(done).astype(np.uint8)

        # obs = torch.from_numpy(obs).float().to(self.device)
        # reward = torch.from_numpy(reward).float()
        obs_ = np.array([o for o in obs]).astype(np.uint8)
        reward_ = np.array([o for o in reward]).astype(np.uint8)
        obs = torch.from_numpy(obs_).float().to(self.device)
        reward = torch.from_numpy(reward_).float()
        return obs, reward, done, info

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_short_term_goal(self, inputs):
        stg = self.venv.get_short_term_goal(inputs)
        stg = torch.from_numpy(stg).float()
        return stg

    def close(self):
        return self.venv.close()