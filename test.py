import habitat
import numpy as np
import cv2
import PIL as Image
import random

config_path = "configs/tasks/pointnav_test.yaml"

# This function is for cv2 showing images
def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

map_size_cm - 2400

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
    params['agent_height'] = 1.25 #     self.args.camera_height * 100
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


def test():
    config = habitat.get_config(config_path)

    config.defrost()
    config.SEED = random.randrange(0, 1000)
    config.SIMULATOR.SEED = random.randrange(0, 1000)
    config.freeze()
    env = habitat.Env(config)
    obs = env.reset()
    
    bgr = transform_rgb_bgr(obs["rgb"])
    cv2.imshow("RGB", bgr)
    cv2.imshow("D", obs["depth"])
    cv2.waitKey(0)

if __name__ == "__main__":
    test()
