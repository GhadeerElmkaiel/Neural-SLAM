import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# color_path = 'RGB/V00P00A00C00.avi'
# depth_path = 'Depth/V00P00A00C00.avi'
# colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc('XVID'), 30, (320,240), 1)
# depthwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc('XVID'), 30, (320,240), 1)
i = 0 
try:
    while True:
        x= input("h")
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        while not depth_frame or not color_frame:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

        img = np.asanyarray(color_frame.get_data())
        img = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

        cv2.imshow('RealSense', img)
        cv2.waitKey(1)
        cv2.imwrite(f'test_imgs/img_{x}_{i}.jpg', img)
        i+=1
        # print(color_frame)
        if not depth_frame or not color_frame:
            continue

finally:
    pipeline.stop()



# # Create a context object. This object owns the handles to all connected realsense devices
# pipeline = rs.pipeline()
# pipeline.start()

# try:
#     while True:
#         # Create a pipeline object. This object configures the streaming camera and owns it's handle
#         frames = pipeline.wait_for_frames()
#         depth = frames.get_depth_frame()
#         if not depth: continue

#         # Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and approximating the coverage of pixels within one meter
#         coverage = [0]*64
#         for y in range(480):
#             for x in range(640):
#                 dist = depth.get_distance(x, y)
#                 if 0 < dist and dist < 1:
#                     coverage[int(x/10)] += 1

#             if y%20 is 19:
#                 line = ""
#                 for c in coverage:
#                     line += " .:nhBXWW"[int(c/25)]
#                 coverage = [0]*64
#                 print(line)

# finally:
#     pipeline.stop()