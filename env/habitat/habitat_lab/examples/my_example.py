import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb
import numpy as np
import matplotlib.pyplot as plt
import quaternion

## Files used from ANS
import env.utils.depth_utils as du
from env.utils.fmm_planner import FMMPlanner
from env.utils.map_builder import MapBuilder
import env.utils.rotation_utils as ru


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"

map_size_cm = 2400 

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
        # self.selem = skimage.morphology.disk(1)     # self.args.obstacle_boundary / self.args.map_resolution
        
        mapper = MapBuilder(params)
        return mapper


def _preprocess_depth(depth):
    depth = depth[:, :, 0]*1
    mask2 = depth > 0.99
    depth[mask2] = 0.

    for i in range(depth.shape[1]):
        depth[:,i][depth[:,i] == 0.] = depth[:,i].max()

    mask1 = depth == 0
    depth[mask1] = np.NaN
    depth = depth*1000.
    return depth


def plot_point(map, point_center, size, borders, color):
    for i in range(size):
        for j in range(size):
            x = (int)(point_center[0]+i-size/2)
            y = (int)(point_center[1]+j-size/2)
            if x >= 0 and x < borders[0] and y >= 0 and y < borders[1]:
                map[x, y]=color


def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    plt.show()
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)


def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def get_map_location(env, resolution):
    bounds = env._sim.pathfinder.get_bounds()
    agent_state = env.sim.get_agent_state()
    agent_pos = agent_state.position

    x = (int)((agent_pos[2]- bounds[0][2])/resolution)
    y = (int)((agent_pos[0]- bounds[0][0])/resolution)
    
    agent_rot = env.sim.get_agent_state().rotation
    axis = quaternion.as_euler_angles(agent_rot)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def get_sim_location(env):
    bounds = env._sim.pathfinder.get_bounds()

    agent_state = env.sim.get_agent_state()
    agent_pos = agent_state.position

    x = -agent_pos[2]
    y = -agent_pos[0]
    
    agent_rot = env.sim.get_agent_state().rotation
    axis = quaternion.as_euler_angles(agent_rot)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    return x, y, o

def example():
    env = habitat.Env(
        config=habitat.get_config("/home/ghadeer/Projects/Neural-SLAM/habitat-lab/configs/tasks/pointnav.yaml")
    )

    print("Environment creation successful")
    observations = env.reset()
    print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
        observations["pointgoal_with_gps_compass"][0],
        observations["pointgoal_with_gps_compass"][1]))
    cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

    resolution = 0.05
    test_map = env._sim.pathfinder.get_topdown_view(resolution, 0)

    recolor_map = np.array(
            [[255, 255, 255], [0, 0, 0]], dtype=np.uint8
    )
    # new_map = np.zeros(test_map.shape)
    # for i in range(test_map.shape[0]):
    #     for j in range(test_map.shape[1]):
    #         if test_map[i,j]:
    #             new_map[i,j] = 1
    #         else:
    #             new_map[i,j] = 0
    # print(new_map)
    # final_map = recolor_map[new_map]
    # final_map = recolor_map[test_map]
    # print(final_map)

    new_map = np.zeros([test_map.shape[0], test_map.shape[1], 3])
    for i in range(test_map.shape[0]):
        for j in range(test_map.shape[1]):
            if test_map[i,j]:
                new_map[i,j]= [0, 0, 0]
            else:
                new_map[i,j]= [255, 255, 255]
    #agent_pos = env.sim.get_agent_state().position
    cv2.imshow("map", new_map)
        
    # print(env._sim.pathfinder.get_bounds())

    bounds = env._sim.pathfinder.get_bounds()
    full_length = bounds[1][0] - bounds[0][0]
    full_weidth = bounds[1][2] - bounds[0][2]
    print("Full Length in meters: ", full_length)
    print("Full Weidth in meters: ", full_weidth)
    print("map shape: ", new_map.shape)
    print("Agent stepping around inside environment.")
    
    mapper = build_mapper()
    mapper.reset_map(map_size_cm)

    old_x = 0
    old_y = 0
    count_steps = 0
    while not env.episode_over:
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.MOVE_FORWARD
            print("action: FORWARD")
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.TURN_LEFT
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.TURN_RIGHT
            print("action: RIGHT")
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.STOP
            print("action: FINISH")
        else:
            print("INVALID KEY")
            continue

        observations = env.step(action)
        count_steps += 1
        # pfrint(observations) 


        # finding robot position in real coordinates and transforming them to map indices

        x, y, o = get_sim_location(env)
        agent_map_x, agent_map_y, o = get_map_location(env, resolution)
        mapper_gt_pose = (x*100.0, y*100.0, np.deg2rad(o))
        # print("x: ", agent_map_x)
        # print("y: ", agent_map_y)
        if agent_map_x < 0:
            agent_map_x = 0
        if agent_map_y < 0:
            agent_map_y = 0
        if agent_map_x >= new_map.shape[0]:
            agent_map_x = new_map.shape[0]-1
        if agent_map_y >= new_map.shape[1]:
            agent_map_y = new_map.shape[1]-1


        # Plot robot's position and old positions
        plot_point(new_map, [agent_map_x, agent_map_y], 3, new_map.shape, [0, 0, 200])
        if count_steps > 1 and (agent_map_x!=old_x or agent_map_y!=old_y):
            plot_point(new_map, [old_x, old_y], 3, new_map.shape, [0, 200, 0])
            # new_map[old_x, old_y]=[0,100,0]
        old_x = agent_map_x
        old_y = agent_map_y
        
        depth = observations["depth"]
        depth = _preprocess_depth(depth)
        fp_proj, full_map, fp_explored, explored_map = \
            mapper.update_map(depth, mapper_gt_pose)
        # print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
            # observations["pointgoal_with_gps_compass"][0],
            # observations["pointgoal_with_gps_compass"][1]))
        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))
        cv2.imshow("gray", observations["depth"])
        
        cv2.imshow("map", new_map)

        print("Projection: \n",fp_proj)
        print("Full map: \n",full_map)
        # semantic_obs = observations["semantic"]
        # semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        # semantic_img.putpalette(d3_40_colors_rgb.flatten())
        # semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        # semantic_img = semantic_img.convert("RGBA")
        # semantic_img = np.array(semantic_img)
        # # print(semantic_img)
        # cv2.imshow("semantic", semantic_img)

    print("Episode finished after {} steps.".format(count_steps))

    if (
        action == HabitatSimActions.STOP
        and observations["pointgoal_with_gps_compass"][0] < 0.2
    ):
        print("you successfully navigated to destination point")
    else:
        print("your navigation was unsuccessful")


if __name__ == "__main__":
    example()
