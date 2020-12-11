import cv2
import numpy as np
import matplotlib.pyplot as pyplot


def draw_local_maps(local_map):
    imgs_1 = local_map[0, :, :, :].cpu().numpy()
    imgs_2 = local_map[1, :, :, :].cpu().numpy()
    cv2.imshow("Proj", imgs_1[0])
    cv2.imshow("Map", imgs_1[1])
    cv2.imshow("Proj2", imgs_2[0])
    cv2.imshow("Map2", imgs_2[1])
