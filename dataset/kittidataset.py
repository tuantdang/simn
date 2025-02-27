# @File      config.py
# @Author    Tuan Dang  
# @Email    tuan.dang@uta.edu or dangthanhtuanit@gmail.com
# Copyright (c) 2024 Tuan Dang, all rights reserved

from os.path import join
from os import listdir
from natsort import natsorted

from lib.utils import voxel_down_sample, crop_frame
import numpy as np
from typing import List
from rich import print # print with co
from torch import tensor, Tensor
from torch import min, max, abs
from numpy.linalg import inv as np_inv
import time
import time
from dataset.dataset import Dataset
from lib.logger import Logger

class KittiDataset(Dataset):
    def __init__(self, cfg, logger:Logger=None):
        super().__init__(cfg, logger)
        self.read_metadata()
        
    def read_metadata(self):
        if self.cfg.dataset.pointcloud != None:
            pc_path = join(self.cfg.dataset.root, self.cfg.dataset.pointcloud)
            self.pc_filenames = natsorted(listdir(pc_path))
            self.pc_count = len(self.pc_filenames)
            if self.cfg.verbose > 0:
                pc_path = join(self.cfg.dataset.root, self.cfg.dataset.pointcloud, self.pc_filenames[0]) # read first pc
                points = self.read_pointcloud(pc_path) # (N,C)
                self.logger.log(1, f'Number of point clouds: {self.pc_count}, point cloud dimension (channels) C = {points.shape[1]}')
            
        if self.cfg.dataset.calib != None:
            calib_path = join(self.cfg.dataset.root, self.cfg.dataset.calib)
            self.calib = self.read_calib(calib_path)
            
        if self.cfg.dataset.pose != None:
            pose_path = join(self.cfg.dataset.root, self.cfg.dataset.pose)
            poses_uncalib = self.read_poses(pose_path)
            if self.cfg.verbose > 0:
                self.logger.log(1, f'Number of GT poses: {len(poses_uncalib)}') 
        if self.cfg.dataset.use_pose:
            gt_poses = self.apply_calib(np.array(poses_uncalib), np_inv(self.calib["Tr"])) # To camera coordinates
            self.gt_pose = tensor(gt_poses, device=self.cfg.device, dtype=self.cfg.dtype.transformation)   
    
    def available(self):
        return self.frame_id < self.pc_count
    
        
    def next_frame(self):
        t1 = time.time()
        self.frame_id += 1
        if self.frame_id >= self.pc_count:
            self.logger.log(1, f'The end of dataset: {self.frame_id+1}/{self.pc_count}')
            return None, None
        #Point cloud
        pc_path = join(self.cfg.dataset.root, self.cfg.dataset.pointcloud, self.pc_filenames[self.frame_id])
        points = self.read_pointcloud(pc_path) # (N,C)
        self.cur_pc = tensor(points, device=self.cfg.device, dtype=self.cfg.dtype.point)
        
        # Point semantic
        if self.cfg.pc.use_point_sematic:
            label_path = join(self.cfg.dataset.root, self.cfg.dataset.label, self.pc_filenames[self.frame_id].replace(".bin", ".label"))
            labels = self.read_label(label_path)   # (N)
            self.cur_point_labels = tensor(labels, device=self.cfg.device, dtype=self.cfg.dtype.index)
        
        #Adaptive threshold based on the current point cloud size
        pc_max_bound, _ = max(self.cur_pc[:, :3], dim=0) # get values [max_x, max_y, max_z]
        pc_min_bound, _ = min(self.cur_pc[:, :3], dim=0) # get values [min_x, min_y, min_z]
        min_x_val = min(abs(pc_max_bound[0]), abs(pc_min_bound[0])) # select the shorter x bound
        min_y_val = min(abs(pc_max_bound[1]), abs(pc_min_bound[1])) # select the shorter y bound
        max_x_y_min_val = max(min_x_val, min_y_val)
        crop_range_val = min(tensor(self.cfg.pc.max_range), 2.0 * max_x_y_min_val).cpu().item()
        # self.logger.log(1, f'max_x_y_min_val: {max_x_y_min_val}, crop_max_range: {crop_range_val}')
        self.logger.log(1, f'FrameId={self.frame_id: 02d}')
        # self.logger.log(1, f'  KittiDataset->nextframe: pc_max_bound={pc_max_bound}, pc_min_bound={pc_min_bound}')
        # self.logger.log(1, f'crop_range_val={crop_range_val} {type(crop_range_val)}')
        
        
        self.voxel_size_pc = round((crop_range_val/self.cfg.pc.max_range)*self.cfg.pc.vs_preprocessing, 2)
        self.vs_reg =  round((crop_range_val/self.cfg.pc.max_range)*self.cfg.pc.vs_reg, 1)
        N0 = self.cur_pc.shape[0]
        self.cur_pc, _ = voxel_down_sample(self.cur_pc, self.voxel_size_pc, self.cfg.verbose)
        N1 = self.cur_pc.shape[0]
        self.cur_pc = crop_frame(self.cur_pc,
                                    self.cfg.pc.min_z, self.cfg.pc.max_z, 
                                    self.cfg.pc.min_range, self.cfg.pc.max_range)
        N2 = self.cur_pc.shape[0]
        
        if self.frame_id == 0:
            self.last_pose = self.cur_pose
            self.cur_pose_guess = self.cur_pose
        else:
            self.cur_pose_guess = self.cur_pose @ self.relative_pose # assume relative_pose is still same or T12=T23 in the next frame
        
        t2 = time.time()
        self.logger.log(1, f'  KittiDataset->next_frame: N0={N0} --down_voxel={self.voxel_size_pc}-> N1={N1}, --cropped-> N2={N2}, processing: {(t2-t1)*1000: .2f} ms')
        return self.frame_id, self.cur_pc[:,:3], None, None # frame_id, pc, rgb, depth


    def read_pointcloud(self, path):
        if ".bin" in path:
            points = np.fromfile(path, dtype=np.float32).reshape((-1, 4))
        return points

    def read_label(self, path):
        labels = np.fromfile(path, dtype=np.uint32).reshape(-1)
        labels = np.array(labels, dtype=np.int32) 
        return labels # label per point

    def read_calib(self, filename):
        calib = {}
        with open(filename, "r") as file:
            lines = file.readlines()
            for line in lines:
                key, content = line.strip().split(":")
                values = [float(v) for v in content.strip().split()]
                pose = np.zeros((4, 4))

                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0

                calib[key] = pose
        return calib

    def read_poses(self, filename):
        poses = []
        with open(filename, 'r') as file:            
            for line in file:
                values = line.strip().split()
                if len(values) < 12: # FIXME: > 12 means maybe it's a 4x4 matrix
                    self.logger.log(1, 'Not a kitti format pose file')
                    return None

                values = [float(value) for value in values]
                pose = np.zeros((4, 4))
                pose[0, 0:4] = values[0:4]
                pose[1, 0:4] = values[4:8]
                pose[2, 0:4] = values[8:12]
                pose[3, 3] = 1.0
                poses.append(pose)
        return poses

    def apply_calib(self, poses:np.ndarray, calib_T_cl:np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame (# T_camera<-lidar)"""
        poses_calib_np = poses.copy()
        for i in range(poses.shape[0]):
            poses_calib_np[i, :, :] = calib_T_cl @ poses[i, :, :] @ np_inv(calib_T_cl) # X = T X T^-1 => XT = TX 

        return poses_calib_np

