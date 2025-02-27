# @Author    Tuan Dang  
# @Email    tuan.dang@uta.edu or dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

from os.path import join
import numpy as np
from typing import List
from rich import print # print with co
from torch import tensor, Tensor
from torch.linalg import inv 
from numpy.linalg import inv as np_inv
import time
from dataset.dataset import Dataset
from lib.logger import Logger
from lib.config import Config
import cv2
import open3d as o3d
from lib.visualizer import Visualizer as Vis
from lib.liealgebra import LieAlgebra as la
import torch
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
import math
from lib.utils import voxel_down_sample, detect_nan, print_object

import numpy as np
from multiprocessing import Pool
from functools import partial
import pyrealsense2 as rs   

class RealsenseCam(Dataset):
    def __init__(self, cfg:Config=None, logger:Logger=None):
        super().__init__(cfg, logger)
        
        
        self.logger = logger
        self.pipeline = rs.pipeline()

        rs_config = rs.rs_config()
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, cfg.camera.fps)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, cfg.camera.fps)

        profile = self.pipeline.start(rs_config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        # Real camera intrinsic
        cfg.camera.fx = float(intr.fx) # Focal length of x
        cfg.camera.fy = float(intr.fy) # Focal length of y
        cfg.camera.cx = float(intr.ppx) # Principle Point Offsey of x (aka. cx)
        cfg.camera.cy = float(intr.ppy) # Principle Point Offsey of y (aka. cy)
      
        # Depth sensor
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = rs_config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        cfg.camera.dscale = 1./depth_sensor.get_depth_scale()

        print_object(cfg.camera)
        # exit()

        self.cfg = cfg
        self.Tx180Neg = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180), dev=self.cfg.device)
        self.skip_first_frames()

    def skip_first_frames(self):
        #Skip some first frames
        for _ in range(30*10):
            self.pipeline.wait_for_frames()

    def available(self): # Query available data
        return True

        
    def next_frame(self): # Get next frame: frame_id, pc, pc_registration
        self.frame_id += 1
        t1 = time.time()
        frameset = self.pipeline.wait_for_frames()
        color_frame = frameset.get_color_frame()
        # depth_frame = frameset.get_depth_frame()

        rgb = np.asanyarray(color_frame.get_data())
        
        # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        #Align
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)
        aligned_depth_frame = frameset.get_depth_frame()
        depth = np.asanyarray(aligned_depth_frame.get_data()) #[w,h]
        str_access_time = f' accsing camera time : {(time.time() - t1)*1000: 0.2f} ms'

        points = self.to_pcd(depth, rgb, self.cfg.camera)
        
        
        # For debuggin
        # colorizer = rs.colorizer()
        # depth_frame_view = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())     #[w,h,3]
        #frame = np.concatenate([color_frame, depth_frame_view], axis=1)
        # cv2.imshow("Realsense D435i", frame)
        # cv2.waitKey(1)  

        # if self.cfg.dataset.cached:
        tpoints = tensor(points, device=self.cfg.device, dtype=self.cfg.dtype.point)
        tpoints[:,:3] = la.transform(tpoints[:,:3], self.Tx180Neg)
        # Vis("Realsene-D435").draw([tpoints], [Vis.red])
        
        # For debugging only
        pc_max_bound, _ = torch.max(tpoints[:, :3], dim=0) # get values [max_x, max_y, max_z]
        pc_min_bound, _ = torch.min(tpoints[:, :3], dim=0) # get values [min_x, min_y, min_z]
        max_bound = pc_max_bound.cpu().numpy()
        min_bound = pc_min_bound.cpu().numpy()
        self.logger.write_range(f'{self.frame_id:05d}, {min_bound[0]:+0.4f}, {max_bound[0]:+0.4f}, {min_bound[1]:+0.4f}, {max_bound[1]:+0.4f}, {min_bound[2]:+0.4f}, {max_bound[2]:+0.4f}')
        
        N0 = tpoints.shape[0]
        tpoints, _ = voxel_down_sample(tpoints, self.cfg.pc.vs_preprocessing, self.cfg.verbose)
        N1 = tpoints.shape[0]
        
        # check z-axis where camera point to scence with negative direction
        if self.cfg.pc.range_checked: # used
            z = tpoints[:, 2]
            mask =  (self.cfg.pc.min_z < z) & (z < self.cfg.pc.max_z)
            tpoints = tpoints[mask]
            N2 = tpoints.shape[0]
        else:
            N2 = N1 # No check for logging
        self.logger.log(1, f'FrameId={self.frame_id}: {str_access_time}')
        self.logger.log(1, f'  TumDataset->next_frame: Original={N0} -downsample-> {N1}, --cropped--> {N2}')
        
        # detect_nan(tpoints)
        return self.frame_id, tpoints, rgb, depth

    def to_pcd(self, depth, #[w,h] 
               rgb, #[w,h,3]
               camera)->np.ndarray:
        h, w = depth.shape
        N = h*w
        points = np.zeros((N,6)) # XYZ-RGB
        depth_scale = camera.dscale
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        i = 0
        for u in range(w): # this process is slow and cached to speedup
            for v in range(h):
                z = depth[v, u]/depth_scale
                points[i, 0] = ((u-cx)*z)/fx # x
                points[i, 1] = ((v-cy)*z)/fy # y
                points[i, 2] = z             # z
                points[i, 3:] = rgb[v, u]/255.0
                i += 1
        return points
