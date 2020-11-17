# Main idea of this code was addapted from 
# https://github.com/devendrachaplot/Neural-SLAM/blob/master/env/habitat/exploration_env.py

import math
import pickle # for noise models loading
import sys
import os

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import skimage.morphology
import torch
import PIL as Image
from torch.nn import functional as F
from torchvision import transforms

from env.utils.map_builder import MapBuilder

# from env.habitat.utils.noisy_actions import CustomActionSpaceConfiguration

import habitat
from habitat import logger

#import env
from env.habitat.utils.pose import *
#import habitat.utils.pose as pu
from env.habitat.utils.supervision import HabitatMaps


from model import get_grid


#TODO   consider the max_depth (the depth image range [0, 1] <=> [0, max_depth])
#       in the testing configiration max_depth = 10
def _preprocess_depth(depth, max_depth=10):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:, i][depth[:,i]== 0.] = depth[:,i].max()

    mask = depth == 0
    depth[mask] = np.NaN
    depth = depth * 100 * max_depth

    return depth

class Neural_SLAM_Env(habitat.RLEnv):

    def __init__(self, args, rank, config_env, dataset):
        if args.visualize:
            plt.ion()
        if args.print_images or args.visualize:
            self.figure, self.ax = plt.subplots(1,2, figsize=(6*16/9, 6),
                                                facecolor="whitesmoke",
                                                num="Thread {}".format(rank))
        self.args = args
        self.rank = rank
        self.num_actions = 4    # number of ations (3) In Active Neural SLAM
        self.dt = 10
        self.timestep = 0


        self.sensor_noise_fwd = \
                pickle.load(open("noise_models/sensor_noise_fwd.pkl", 'rb'))
        self.sensor_noise_right = \
                pickle.load(open("noise_models/sensor_noise_right.pkl", 'rb'))
        self.sensor_noise_left = \
                pickle.load(open("noise_models/sensor_noise_left.pkl", 'rb'))

        #TODO def noisy actions 

        # config_env.defrost()
        # config_env.SIMULATOR.ACTION_SPACE_CONFIG = "CustomActionSpaceConfiguration"
        # config_env.freeze()

        super().__init__(config_env, dataset)
        
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                    args.frame_width),
                                                dtype='uint8')

        self.mapper = self.build_mapper()
        
        self.episode_no = 0
        
        self.res = transforms.Compose([transforms.ToPILImage(),
                    transforms.Resize((args.frame_height, args.frame_width),
                                      interpolation = Image.Image.NEAREST)])
        self.scene_name = None
        self.maps_dict = {}

    # Function To Randomize The Environment
    def randomize_env(self):
        self._env._episode_iterator._shuffle_iterator()

    def save_trajectory_data(self):
        if "replica" in self.scene_name:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-3]+"/"
        else:
            folder = self.args.save_trajectory_data + "/" + \
                        self.scene_name.split("/")[-1].split(".")[0]+"/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        filepath = folder+str(self.episode_no)+".txt"
        with open(filepath, "w+") as f:
            f.write(self.scene_name+"\n")
            for state in self.trajectory_states:
                f.write(str(state)+"\n")
            f.flush()

    def save_position(self):
        self.agent_state = self._env.sim.get_agent_state()
        self.trajectory_states.append([self.agent_state.position,
                                       self.agent_state.rotation])

    def reset(self):
        args = self.args
        self.episode_no += 1
        self.timestep = 0
        self._previous_action = None
        self.trajectory_states = []

        # Randomize The Environment
        if args.randomize_env_every > 0:
            if np.mod(self.episode_no, args.randomize_env_every) ==0:
                self.randomize_env()
        
        # Get Gound Truth Map
        self.explorable_map = None
        while self.explorable_map is None:
            obs = super().reset()
            full_map_size = args.map_size_cm//args.map_resolution
            self.explorable_map = self._get_gt_map(full_map_size)
        self.prev_explored_area = 0.

        # Preprocess observations
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))
        state = rgb.transpose(2, 0, 1)
        depth = _preprocess_depth(obs['depth'])

        # Initialize map and pose
        self.map_size_cm = args.map_size_cm
        self.mapper.reset_map(self.map_size_cm)
        self.curr_loc = [self.map_size_cm/100.0/2.0,
                         self.map_size_cm/100.0/2.0, 0.0]
        self.curr_loc_gt = self.curr_loc
        self.last_loc_gt = self.curr_loc_gt
        self.last_loc = self.curr_loc
        self.last_sim_location = self.get_sim_location()

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))

        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
            self.mapper.update_map(depth, mapper_gt_pose)

        # Initialize variables
        self.scene_name = self.habitat_env.sim.config.sim_cfg.scene.id
        self.visited = np.zeros(self.map.shape)
        self.visited_vis = np.zeros(self.map.shape)
        self.visited_gt = np.zeros(self.map.shape)
        self.collison_map = np.zeros(self.map.shape)
        self.col_width = 1

        # Set info
        self.info = {
            'time': self.timestep,
            'fp_proj': fp_proj,
            'fp_explored': fp_explored,
            'sensor_pose': [0., 0., 0.],
            'pose_err': [0., 0., 0.],
        }

        self.save_position()
        return state, self.info

    def step(self, action):
        args = self.args
        self.timestep += 1

        #TODO
        # Add noisy actions

        self.last_loc = np.copy(self.curr_loc)
        self.last_loc_gt = np.copy(self.curr_loc_gt)
        self._previous_action = action

        #TODO
        # For noisy actions
        '''
        if args.noisy_actions:
            obs, rew, done, info = super().step(noisy_action)
        else:
            obs, rew, done, info = super().step(action)
        '''


        obs, rew, done, info = super().step(action)
        rgb = obs['rgb'].astype(np.uint8)
        self.obs = rgb # For visualization
        if self.args.frame_width != self.args.env_frame_width:
            rgb = np.asarray(self.res(rgb))


        state = rgb.transpose(2, 0, 1)

        depth = _preprocess_depth(obs['depth'])

        # Get base sensor and ground-truth pose
        dx_gt, dy_gt, do_gt = self.get_gt_pose_change()
        dx_base, dy_base, do_base = self.get_base_pose_change(
                                        action, (dx_gt, dy_gt, do_gt))

        # self.curr_loc = pu.get_new_pose(self.curr_loc,
        #                        (dx_base, dy_base, do_base))

        # self.curr_loc_gt = pu.get_new_pose(self.curr_loc_gt,
        #                        (dx_gt, dy_gt, do_gt))

        self.curr_loc = get_new_pose(self.curr_loc,
                               (dx_base, dy_base, do_base))

        self.curr_loc_gt = get_new_pose(self.curr_loc_gt,
                               (dx_gt, dy_gt, do_gt))

        if not args.noisy_odometry:
            self.curr_loc = self.curr_loc_gt
            dx_base, dy_base, do_base = dx_gt, dy_gt, do_gt

        # Convert pose to cm and degrees for mapper
        mapper_gt_pose = (self.curr_loc_gt[0]*100.0,
                          self.curr_loc_gt[1]*100.0,
                          np.deg2rad(self.curr_loc_gt[2]))


        # Update ground_truth map and explored area
        fp_proj, self.map, fp_explored, self.explored_map = \
                self.mapper.update_map(depth, mapper_gt_pose)


        # Update collision map
        if action == 1:
            x1, y1, t1 = self.last_loc
            x2, y2, t2 = self.curr_loc
            if abs(x1 - x2)< 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                self.col_width = min(self.col_width, 9)
            else:
                self.col_width = 1

            dist = get_l2_distance(x1, x2, y1, y2)
            # dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < args.collision_threshold: #Collision
                length = 2
                width = self.col_width
                buf = 3
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + \
                                        (j-width//2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - \
                                        (j-width//2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r*100/args.map_resolution), \
                               int(c*100/args.map_resolution)
                        [r, c] = threshold_poses([r, c],
                                    self.collison_map.shape)
                        # [r, c] = pu.threshold_poses([r, c],
                        #             self.collison_map.shape)
                        self.collison_map[r,c] = 1

        # Set info
        self.info['time'] = self.timestep
        self.info['fp_proj'] = fp_proj
        self.info['fp_explored']= fp_explored
        self.info['sensor_pose'] = [dx_base, dy_base, do_base]
        self.info['pose_err'] = [dx_gt - dx_base,
                                 dy_gt - dy_base,
                                 do_gt - do_base]

        """
        if self.timestep%args.num_local_steps==0:
            area, ratio = self.get_global_reward()
            self.info['exp_reward'] = area
            self.info['exp_ratio'] = ratio
        else:
            self.info['exp_reward'] = None
            self.info['exp_ratio'] = None
        """

        self.info['exp_reward'] = None
        self.info['exp_ratio'] = None

        self.save_position()

        if self.info['time'] >= args.max_episode_length:
            done = True
            if self.args.save_trajectory_data != "0":
                self.save_trajectory_data()
        else:
            done = False

        return state, rew, done, self.info


    
    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_gt_pose_change(self):
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        # dx, dy, do = pu.get_rel_pose_change(curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do


    def get_base_pose_change(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        if action == 1: ## Forward
            x_err, y_err, o_err = self.sensor_noise_fwd.sample()[0][0]
        elif action == 3: ## Right
            x_err, y_err, o_err = self.sensor_noise_right.sample()[0][0]
        elif action == 2: ## Left
            x_err, y_err, o_err = self.sensor_noise_left.sample()[0][0]
        else: ##Stop
            x_err, y_err, o_err = 0., 0., 0.

        x_err = x_err * self.args.noise_level
        y_err = y_err * self.args.noise_level
        o_err = o_err * self.args.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)




    def build_mapper(self):
        params = {}
        params['frame_width'] = self.args.env_frame_width
        params['frame_height'] = self.args.env_frame_height
        params['fov'] =  self.args.hfov
        params['resolution'] = self.args.map_resolution
        params['map_size_cm'] = self.args.map_size_cm
        params['agent_min_z'] = 25
        params['agent_max_z'] = 150
        params['agent_height'] = self.args.camera_height * 100
        params['agent_view_angle'] = 0
        params['du_scale'] = self.args.du_scale
        params['vision_range'] = self.args.vision_range
        params['visualize'] = self.args.visualize
        params['obs_threshold'] = self.args.obs_threshold
        self.selem = skimage.morphology.disk(self.args.obstacle_boundary /
                                             self.args.map_resolution)
        mapper = MapBuilder(params)
        return mapper

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
    
    def get_spaces(self):
        return self.observation_space, self.action_space

    # Get Map from Simulator
    def _get_gt_map(self, full_map_size):
        self.scene_name = self.habitat_env.sim.config.sim_cfg.scene.id #SCENE
        logger.error('Computing map for %s', self.scene_name)

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            logger.error("Invalid map: {}/{}".format(
                            self.scene_name, self.episode_no))
            return None

        agent_y = self._env.sim.get_agent_state().position.tolist()[1]*100.
        sim_map = self.map_obj.get_map(agent_y, -50., 50.0)

        sim_map[sim_map > 0] = 1.

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin/100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max/100. - self.map_obj.origin/100.

        map_size = sim_map.shape
        scale = 2.
        grid_size = int(scale*max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[(grid_size - map_size[0])//2:
                 (grid_size - map_size[0])//2 + map_size[0],
                 (grid_size - map_size[1])//2:
                 (grid_size - map_size[1])//2 + map_size[1]] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale) \
                             * map_size[1] * 1. / map_size[0],
                    (y - range_y/2.) * 2. / (range_y * scale),
                    180.0 + np.rad2deg(o)
                ]])

        else:
            st = torch.tensor([[
                    (x - range_x/2.) * 2. / (range_x * scale),
                    (y - range_y/2.) * 2. / (range_y * scale) \
                            * map_size[0] * 1. / map_size[1],
                    180.0 + np.rad2deg(o)
                ]])

        rot_mat, trans_mat = get_grid(st, (1, 1,
            grid_size, grid_size), torch.device("cpu"))

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[(full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size,
                        (full_map_size - grid_size)//2:
                        (full_map_size - grid_size)//2 + grid_size] = \
                                rotated[0,0]
        else:
            episode_map = rotated[0,0,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size,
                              (grid_size - full_map_size)//2:
                              (grid_size - full_map_size)//2 + full_map_size]



        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.

        return episode_map


    
    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_reward(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return 0.

    def get_done(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        return False

    def get_info(self, observations):
        # This function is not used, Habitat-RLEnv requires this function
        info = {}
        return info