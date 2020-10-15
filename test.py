import habitat
import numpy as np
import cv2
import PIL as Image
import random

config_path = "configs/tasks/pointnav_test.yaml"


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

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
