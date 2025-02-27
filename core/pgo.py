# @author    Tuan Dang   
# Email: tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import gtsam
from gtsam import noiseModel, Pose3, NonlinearFactorGraph, Values, LevenbergMarquardtOptimizer
import numpy as np 
    
class PoseGraphOptimization:
    def __init__(self, cfg=None, sigma=1.0e-4):
        self.cfg = cfg
        self.sigma = sigma
        self.graph = NonlinearFactorGraph()
        self.initials = Values()
        
    def add_prior(self, 
                  id:int,
                  pose: np.ndarray, # [4,4]
                  cov: np.ndarray=None, # [6,6]
                  ):
        
        priorMean = Pose3(pose)
        if cov is None:
            priorNoise = noiseModel.Diagonal.Sigmas([self.sigma]*6)
        else:
            priorNoise = gtsam.noiseModel.Gaussian.Covariance(cov)
    
        self.graph.add(gtsam.PriorFactorPose3(id, priorMean, priorNoise))
        # print(priorNoise)
        
    def add_odom(   self,
                    id_from:int,
                    id_to: int, 
                    odom: np.ndarray, # [4,4]
                    cov: np.ndarray=None, # [6,6]
                  ):
        
        if cov is None:
            noise = noiseModel.Diagonal.Sigmas([self.sigma]*6)
        else:
            noise = gtsam.noiseModel.Gaussian.Covariance(cov)
    
        self.graph.add(gtsam.BetweenFactorPose3(id_from, id_to, Pose3(odom), noise)) # 0,1 .. 1,2..
    
    def add_init(self, 
                 id : int,
                 init_pose: np.ndarray # [4,4]
                 ):
        self.initials.insert(id, Pose3(init_pose))
        
    def optimize(self):
        estimations = LevenbergMarquardtOptimizer(self.graph, self.initials).optimize()
        keys = estimations.keys()
        N = estimations.size() # Number of poses
        est_poses = np.zeros((N, 4, 4))
        for i in range(N):
            pose3 = estimations.atPose3(keys[i])
            est_poses[i,...] = pose3.matrix()
        return est_poses
    
if __name__ == "__main__":
    pgo = PoseGraphOptimization()
    pgo.add_prior(np.eye(4))