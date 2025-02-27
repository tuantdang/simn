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
from lib.utils import voxel_down_sample, detect_nan

import numpy as np
from multiprocessing import Pool
from functools import partial
from natsort import natsorted
from os import listdir

class UtaDataset(Dataset):
    def __init__(self, cfg:Config=None, logger:Logger=None):
        super().__init__(cfg, logger)
        
        self.cfg = cfg
        self.logger = logger
        self.dataset_path = join(cfg.dataset.root, cfg.dataset.sub_path)
        self.rgb_path = join(self.dataset_path, 'rgb')
        self.depth_path = join(self.dataset_path, 'depth')
        self.rgb_files = natsorted(listdir(self.rgb_path))
        self.depth_files = natsorted(listdir(self.depth_path))
        self.pc_count = len(self.depth_files)
        self.Tx180Neg = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180), dev=self.cfg.device) # Get transformation rotating of Ox 180 deg
        self.cached = self.cfg.dataset.cached
        if self.cached:
            self.build_cache()
        
        # For debugging    
        os.makedirs(f'{logger.dir}/rgb_depth', exist_ok=True)
    
    
    def available(self):
        return self.frame_id < self.pc_count
    
    def save_rgbd_figure(self, rgb, depth, frame_id):
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgb)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(depth)
        plt.savefig(f'{self.logger.dir}/rgb_depth/{frame_id:04d}.png')
        # plt.show()
    
    def get_frame(self, frame_id) -> np.ndarray:
        depth_file = join(self.depth_path, self.depth_files[frame_id])
        rgb_file   = join(self.rgb_path, self.rgb_files[frame_id])
        rgb = np.load(rgb_file) # [480, 640, 3]
        depth = np.load(depth_file) # [480, 640]
        
        if self.cfg.dataset.multi_processing: # not used
            points = self.to_pcd_multi_processing(depth, rgb, self.cfg.camera)
        else:
            points = self.to_pcd(depth, rgb, self.cfg.camera)
        return points, rgb, depth
    
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
        # image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        for u in range(w): # this process is slow and cached to speedup
            for v in range(h):
                z = depth[v, u]/depth_scale
                points[i, 0] = ((u-cx)*z)/fx # x
                points[i, 1] = ((v-cy)*z)/fy # y
                points[i, 2] = z             # z
                points[i, 3:] = rgb[v, u]/255.0
                i += 1
        return points
    
    def to_rgb_depth(self, points: np.ndarray, # [N,6]
                     camera # camera information
        ):
        depth_scale = camera.dscale
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        w, h = camera.width, camera.height
        depth = np.full((h,w), np.inf)
        rgb = np.full((h,w,3), 1.0)
        N = points.shape[0]
        for i in range(N):
            x,y,z = points[i, :3]
            if z > 0:
                d = z*depth_scale
                u = math.floor((x*fx)/z + cx) # horizontal axis
                v = math.floor((y*fy)/z + cy) # vertical axis
                depth[v,u] = d
                rgb[v, u] =  points[i, 3:] # depend on the quality of depth (i.e depth has more z=0)
            
        return rgb, depth
        
    
    def check_cached_dir(self, dir):
        if os.path.isdir(dir): # exist
            files = os.listdir(dir)
            if len(files) != self.pc_count:
                self.logger.log(1, f'Cache at {dir} is invalid, need to be rebuilt!')
                # self.logger.log(1, f'Removed the cached directory, need to run again to build it.')
                shutil.rmtree(dir) # remove non-empty directory
                exit()
            else:
                print(f'check_cached_dir: {dir}, files = {len(files)}')
                return True
        else:
            return False
        return True

    def build_cache(self): # each file ~14.7MB -> 2856 frames ~ 42GB. Originally: depth+rgb ~ 1.7GB
        # Check whether cache is build previously
        dir = f'{self.dataset_path}/cached'
        dir_pcd = f'{dir}/pcd'
        dir_rgb = f'{dir}/rgb'
        dir_depth = f'{dir}/depth'
        
        if self.check_cached_dir(dir_pcd) and self.check_cached_dir(dir_rgb) and self.check_cached_dir(dir_depth):
            self.logger.log(1, f'Cache at {dir} was built previously! Just use it!')
        else: # not built -> now build
            self.logger.log(1, f'Building cache for {self.pc_count} frame')
            os.makedirs(dir, exist_ok=True)
            os.makedirs(dir_pcd, exist_ok=True)
            os.makedirs(dir_rgb, exist_ok=True)
            os.makedirs(dir_depth, exist_ok=True)
            for frame_id in tqdm(range(self.pc_count), desc='Building cache...'):
                points, rgb, depth = self.get_frame(frame_id)
                np.save(f'{dir_pcd}/{frame_id:05d}.npy', points)
                np.save(f'{dir_rgb}/{frame_id:05d}.npy', rgb)
                np.save(f'{dir_depth}/{frame_id:05d}.npy', depth)
            self.logger.log(1, f"Build cache for {self.pc_count} frames done!")
    

    
    def process_chunk(chunk_data):
        depth_chunk, rgb_chunk, start_x, start_y, fx, fy, cx, cy = chunk_data
        h, w = depth_chunk.shape
        points = np.zeros((h*w,6))
        i = 0
        for v in range(h):
            for u in range(w):
                z = depth_chunk[v, u]
                points[i, 0] = ((u-cx)*z)/fx # x
                points[i, 1] = ((v-cy)*z)/fy # y
                points[i, 2] = z             # z
                points[i, 3:] = rgb_chunk[v, u]/255.0
                i += 1
        return points
    
    def to_pcd_multi_processing(self, depth, rgb, camera):
        fx, fy = camera.fx, camera.fy
        cx, cy = camera.cx, camera.cy
        chunk_size = self.cfg.dataset.chunk_size
        chunks = []
        depth = depth/self.cfg.camera.dscale
        for start_y in range(0, depth.shape[0], chunk_size):
            for start_x in range(0, depth.shape[1], chunk_size):
                end_y = min(start_y + chunk_size, depth.shape[0])
                end_x = min(start_x + chunk_size, depth.shape[1])
                depth_chunk = depth[start_y:end_y, start_x:end_x]
                rgb_chunk = rgb[start_y:end_y, start_x:end_x]
                chunks.append((depth_chunk, rgb_chunk, start_x, start_y, fx, fy, cx, cy))

        # Process in parallel
        with Pool() as pool:
            chunks = pool.map(UtaDataset.process_chunk, chunks)
            pc = torch.empty((0, 6)).to(self.cfg.device, dtype=self.cfg.dtype.point)
            # Convert to point clouds
            for chunk in chunks:
                pc = torch.cat([pc, tensor(chunk).to(self.cfg.device, dtype=self.cfg.dtype.point)], dim=0)
        return pc.cpu().numpy()
    
    def next_frame(self) -> Tensor:
        self.frame_id += 1
        if self.frame_id >= self.pc_count:
            if self.logger != None:
                self.logger.log(1, f'The end of dataset: {self.frame_id+1}/{self.pc_count}')
            return None, None
       
        if not self.cached: # single thread: 468 ms, multi-processing: 260 ms
            t1 = time.time()
            points, rgb, depth = self.get_frame(self.frame_id)
            t2 = time.time()
            print(f'Not cached: {(t2-t1)*1000:0.1f} ms')
        else: # cached ~ 1.2 ms
            t1 = time.time()
            points = np.load(f'{self.dataset_path}/cached/pcd/{self.frame_id:05d}.npy')
            rgb = np.load(f'{self.dataset_path}/cached/rgb/{self.frame_id:05d}.npy')
            depth = np.load(f'{self.dataset_path}/cached/depth/{self.frame_id:05d}.npy')
            t2 = time.time()
            str_access_time = f' cached_time={(t2-t1)*1000:0.1f} ms'
            
        # For debugging: save rgb,depth every second
        # if self.frame_id % self.cfg.camera.fps == 0:
        if True:
            # rgb, depth = self.to_rgb_depth(points, self.cfg.camera)ge
            self.save_rgbd_figure(rgb, depth, self.frame_id)
        
        # if self.cfg.dataset.cached:
        tpoints = tensor(points, device=self.cfg.device, dtype=self.cfg.dtype.point)
        tpoints[:,:3] = la.transform(tpoints[:,:3], self.Tx180Neg)
        
        # Vis("Dataset").draw([tpoints], [Vis.red]); exit()
        
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


# if __name__ == '__main__':
    # pass
    
    