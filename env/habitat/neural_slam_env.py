# Main idea of this code was addapted from https://github.com/devendrachaplot/Neural-SLAM/blob/master/env/habitat/exploration_env.py

import habitat


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

class Neural_SLAM_env(habitat.RLEnv):

    def __init__(self, args, rank, config_env, dataset):
        #TODO def noisy actions 
        #TODO addapt from exploration_env

        self.args = args
        self.rank = rank


        super.__init__(config_env, dataset)