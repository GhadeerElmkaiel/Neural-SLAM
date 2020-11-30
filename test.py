
import os

from collections import deque

import habitat
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import logging
import gym


import cv2
import matplotlib.pyplot as plt
import PIL as Image
import random

from env import make_vec_envs

from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module
from utils.optimization import get_optimizer
from utils.storage import GlobalRolloutStorage, FIFOMemory

import algo

import pyrealsense2 as rs

from arguments import get_args


args = get_args()

## ------------------- Testing Functions ------------------------

args.num_processes = 2
# args.split = 'val'
args.train_slam = 0
args.load_slam = 'pretrained_models/model_best.slam'
args.map_size_cm = 5000
args.task_config = 'tasks/pointnav_test.yaml'
args.seed = 5



## --------------------------------------------------------------


# Initializing random seeds
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


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


# Get boundaries of the local map
def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


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

    ##########################################################
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    img = np.asanyarray(color_frame.get_data())
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', img)
    cv2.waitKey(1)
    ##########################################################

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

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))


    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    
    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    # Initialize full_map and full_pose
    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()

    # Global policy observation space
    g_observation_space = gym.spaces.Box(0, 1,
                                         (8,
                                          local_w,
                                          local_h), dtype='uint8')

    # Global policy action space
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

    # Local policy observation space
    l_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Local and Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size
    g_hidden_size = args.global_hidden_size

    

    # slam
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(),
                                   args.slam_optimizer)


    '''
    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    # Local policy
    l_policy = Local_IL_Policy(l_observation_space.shape, envs.action_space.n,
                               recurrent=args.use_recurrent_local,
                               hidden_size=l_hidden_size,
                               deterministic=args.use_deterministic_local).to(device)
    local_optimizer = get_optimizer(l_policy.parameters(),
                                    args.local_optimizer)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      1).to(device)
    
    
    slam_memory = FIFOMemory(args.slam_memory_size)
    '''


    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)

    if not args.train_slam:
        nslam_module.eval()
        
    '''
    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if not args.train_global:
        g_policy.eval()

    if args.load_local != "0":
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)

    if not args.train_local:
        l_policy.eval()
    '''

    # Predict map from frame 1:
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx
         in range(num_scenes)])
    ).float().to(device)

    _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
        nslam_module(obs, obs, poses, local_map[:, 0, :, :],
                     local_map[:, 1, :, :], local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    
    global_input = torch.zeros(num_scenes, 8, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    
    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)


    imgs_1 = local_map[0, :, :, :].cpu().numpy()
    imgs_2 = local_map[1, :, :, :].cpu().numpy()
    
    obs_all = _process_obs_for_display(obs)

    # fig, axis = plt.subplots(1, 3)
    # axis[0].imshow(obs_all[0])
    # axis[1].imshow(imgs_1[0], cmap='gray')
    # axis[2].imshow(imgs_1[1], cmap='gray')
    cv2.imshow("Camer", transform_rgb_bgr(obs_all[0]))
    cv2.imshow("Proj", imgs_1[0])
    cv2.imshow("Map", imgs_1[1])


    cv2.imshow("Camer2", transform_rgb_bgr(obs_all[1]))
    cv2.imshow("Proj2", imgs_2[0])
    cv2.imshow("Map2", imgs_2[1])

    action = 1
    while action!= 4:
        k = cv2.waitKey(0)
        if k == 119:
            action = 1
            action_2 = 1
        elif k == 100:
            action = 3
            action_2 = 1
        elif k == 97:
            action = 2
            action_2 = 2
        elif k == 102:
            action = 4
            break
        else:
            action = 1

        last_obs = obs.detach()

        obs, rew, done, infos = envs.step(torch.from_numpy(np.array([action, action_2])))
        
        obs_all = _process_obs_for_display(obs)
        cv2.imshow("Camer", transform_rgb_bgr(obs_all[0]))
        cv2.imshow("Camer2", transform_rgb_bgr(obs_all[1]))

        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
                in range(num_scenes)])
        ).float().to(device)

        _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
            nslam_module(last_obs, obs, poses, local_map[:, 0, :, :],
                            local_map[:, 1, :, :], local_pose, build_maps=True)

        imgs_1 = local_map[0, :, :, :].cpu().numpy()
        imgs_2 = local_map[1, :, :, :].cpu().numpy()
        cv2.imshow("Proj", imgs_1[0])
        cv2.imshow("Map", imgs_1[1])
        cv2.imshow("Proj2", imgs_2[0])
        cv2.imshow("Map2", imgs_2[1])




    # plt.show()

    print("\n\nDone\n\n")
    # input("Press Enter to finish")
    #my_env = Neural_SLAM_Env()

if __name__ == "__main__":
    test()
