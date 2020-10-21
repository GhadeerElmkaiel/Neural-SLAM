
import os

import habitat
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import logging

import cv2
import matplotlib.pyplot as plt
import PIL as Image
import random

from env import make_vec_envs
# For test
from env.habitat.neural_slam_env import Neural_SLAM_Env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.config.default import get_config as cfg_env
from env.habitat.habitat_lab.habitat.core.vector_env import VectorEnv

from arguments import get_args

## ------------------- Testing Functions ------------------------

def make_env_fn(args, config_env, rank):
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()

    env = Neural_SLAM_Env(args=args, rank=rank,
                          config_env=config_env, dataset=dataset
                          )

    env.seed(rank)
    return env

## --------------------------------------------------------------


args = get_args()

# config_path = "/home/ghadeer/Projects/Neural-SLAM/Neural-SLAM/configs/tasks/pointnav_test.yaml"
config_path = "configs/tasks/pointnav_test.yaml"

# This function is for cv2 showing images
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

map_size_cm = 2400

# Prepare parameters for the mapper "MapBuilder" object
def build_mapper():
    params = {}
    params['frame_width'] = 256 #       self.args.env_frame_width
    params['frame_height'] = 256 # s    elf.args.env_frame_height
    params['fov'] = 90.0 #              self.args.hfov
    params['resolution'] = 5 #          self.args.map_resolution
    params['map_size_cm'] = map_size_cm #      self.args.map_size_cm
    params['agent_min_z'] = 25
    params['agent_max_z'] = 150
    params['agent_height'] = 1.25 * 100 #     self.args.camera_height * 100
    params['agent_view_angle'] = 0
    params['du_scale'] = 2 #            self.args.du_scale
    params['vision_range'] = 64 #       self.args.vision_range
    params['visualize'] = 0 #           self.args.visualize
    params['obs_threshold'] = 1 #       self.args.obs_threshold
    
    mapper = MapBuilder(params)
    return mapper


# Function for processing the depth image befor using it
def _preprocess_depth(depth):
    pass


def _process_obs_for_display(obs):
    obs_np = obs.cpu().numpy()
    obs_ = []
    for o in obs_np:
        shp = o.shape
        obs_2 = o.reshape([3, -1])
        obs_3 = obs_2.ravel(order='F').reshape(shp[-1], shp[-1], 3)
        obs_.append(obs_3)
    return np.array(obs_)/256.


def test():

    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level= logging.INFO)
    print("Dumping at {}".format(log_dir))
    logging.info(args)

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()
    obs_np = obs.cpu().numpy()
    # envs = VectorEnv(
    #     make_env_fn= make_env_fn,
    #     env_fn_args= tuple(
    #         tuple(
    #             zip([args], [config_env], [0])
    #         )
    #     ),
    # )

    # my_env = Neural_SLAM_Env(args=args, rank=0,
                        #   config_env=config_env, dataset=dataset
                        #   )

    # my_env.seed(0)
    obs_all = _process_obs_for_display(obs)
    
    # obs_1 = obs_1.reshape(128, 128, 3)
    # obs_np = obs_np.reshape(shp[0], shp[-2], shp[-1], -1)
    # cv2.imshow("First rgb", obs_np[0])
    # cv2.imshow("Second rgb", obs_np[1])
    plt.imshow(obs_all[0])
    plt.show()
    plt.imshow(obs_all[1])
    plt.show()
    print("\n\nDone\n\n")
    input("Press Enter to finish")
    #my_env = Neural_SLAM_Env()

if __name__ == "__main__":
    test()
