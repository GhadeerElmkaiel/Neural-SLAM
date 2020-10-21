import os

import habitat
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import logging

import cv2
import PIL as Image
import random

# For test
from env.habitat.neural_slam_env import Neural_SLAM_Env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.config.default import get_config as cfg_env
from env.habitat.habitat_lab.habitat.core.vector_env import VectorEnv

from arguments import get_args


args = get_args()

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

def main():
    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_lab/configs/" + args.task_config])
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.freeze()

    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    config_env = cfg_env(config_paths=
                         ["env/habitat/habitat_lab/configs/" + args.task_config])
    config_env.defrost()
    config_env.DATASET.CONTENT_SCENES = scenes

    gpu_id = 0
    config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

    agent_sensors = []
    agent_sensors.append("RGB_SENSOR")
    agent_sensors.append("DEPTH_SENSOR")

    config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

    config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
    config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

    config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

    config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
    config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
    config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
    config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

    config_env.SIMULATOR.TURN_ANGLE = 10
    config_env.DATASET.SPLIT = args.split

    config_env.freeze()

    env = make_env_fn(args, config_env, 0)
    obs, inf = env.reset()
    print("done")

if __name__ == "__main__":
    main()
