# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import torch
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np
import sys
import os
import shutil


# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from lib.liealgebra import LieAlgebra as la
from lib.visualizer import Visualizer as Vis

import seaborn as sns
n_colors = 10
# colors_ori = sns.color_palette("flare", n_colors)
# colors = np.array(np.asanyarray(colors_ori)*255, dtype=np.uint8)
colors = np.array(
    [   [205, 92, 92],      # 0: Red
        [255, 105, 180],    # 1: Pink
        [255, 127, 80],     # 2: Coral
        [255, 165, 0],      # 3: Orange
        [255, 215, 0],      # 4: Gold
        [255, 255, 0],      # 5: Yellow
        [189, 183, 107],    # 6: Dark Khakhi    
        [238, 130, 238],    # 7: Violet
        [255, 0, 255],      # 8: Magenta    
        [138, 43, 226],     # 9: Blue Violet
    ]
)


from lib.config import Config

# Dataset
# root_dir = '/home/tuandang/workspace/datasets/uta'
# scene_name1 = 'lab1_demo'
# id0 = 3382

ds = 'config/tum_fr3_office.yaml'
from lib.logger import Logger
from torch import nn
from dataset.tumdataset import TumDataset
cfg = Config(path=f'{ds}').config()
logger = Logger(cfg, print_on=True)
dataset = TumDataset(cfg, logger)

print(dataset.dataset_path)
root_dir = dataset.dataset_path
dataset_name = cfg.dataset.name
scene_name = cfg.dataset.sub_path
id0 = 0

use_mask = False 
if dataset_name == 'uta':
    use_mask = True
    

def to_pcd(image, depth, mask):
    cam = cfg.camera
    w, h = cam.width, cam.height
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy
    dscale = cam.dscale
    N = w*h
    points = np.zeros((N, 6)) # XYZ-RGB
    i = 0
    for u in range(w): # this process is slow and cached to speedup
        for v in range(h):
            if mask[v,u] == 1:
                z = depth[v, u] / dscale
                points[i,0] = ((u-cx)*z)/fx # x
                points[i,1] = ((v-cy)*z)/fy # y
                points[i,2] = z             # z
                points[i,3:] = image[v, u]/255.0
                i += 1
    points = points[:i,:]
    tpoints = torch.tensor(points, dtype=torch.float).clone()
    T = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180), dev='cpu')
    tpoints[:,:3] = la.transform(tpoints[:,:3], T)
    min_z, max_z = -7.5, -0.5
    z = tpoints[:, 2]
    mask =  (min_z < z) & (z < max_z)
    tpoints = tpoints[mask]
    points = tpoints[:,:3].numpy()
    
    npoints = points.shape[0]
    y_mean = np.sum(points[:,1])
    # print(f'npoints = {npoints}, y_mean = {y_mean/npoints}, sum={y_mean}')
    return y_mean, npoints

segments_dir = f'./log/uta/{scene_name}/segments'
# shutil.rmtree(segments_dir)
os.makedirs(segments_dir, exist_ok=True)


# Load the SAM model
if use_mask:
    sam_checkpoint = "/home/tuandang/workspace/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"  # Options: vit_h, vit_l, vit_b
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

def generate_pc(id, scene_name):
    saved_dir = f'{root_dir}'
    if dataset_name == 'uta':
        rgb_dir = f'{saved_dir}/rgb'
        depth_dir = f'{saved_dir}/depth'
        rgb_fn = f'{rgb_dir}/{id:04d}.npy'
        depth_fn = f'{depth_dir}/{id:04d}.npy'
    elif dataset_name == 'tum':
        rgb_dir = f'{saved_dir}/cached/rgb'
        depth_dir = f'{saved_dir}/cached/depth'
        rgb_fn = f'{rgb_dir}/{id:05d}.npy'
        depth_fn = f'{depth_dir}/{id:05d}.npy'
   
    rgb = np.load(rgb_fn)
    depth = np.load(depth_fn)
    image = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Generate all masks
    
    if use_mask:
        masks = mask_generator.generate(image)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        mask_overlay = np.ones_like(image, dtype=np.uint8)
        for i, mask in enumerate(masks):
            mask_overlay[mask["segmentation"]] = colors[i%n_colors]

        # Find floor mask    
        num_masks = len(masks)
        max_segments = 20
        print(f'number mask = {num_masks}')
        y = np.ones((max_segments,))*np.inf
        for i, mask in enumerate(masks):
            if i >= max_segments:
                break
            y_mean, n_points = to_pcd(image, depth, mask["segmentation"])
            y[i] = y_mean

        idx = np.argmin(y)
        print(y)
        mask = masks[idx]["segmentation"]
        

        #lab4
        # mask = np.zeros_like(masks[0]["segmentation"])
        # for i in range(3):
        #     mask |= masks[i]["segmentation"]


        mask_overlay[mask] = np.array([255, 0, 0])
        binary_mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = contours[-1].reshape(-1,2)
        

        # Blend the mask with the original image
        alpha = 0.0
        blended = cv2.addWeighted(image, alpha, mask_overlay, 1-alpha, 0)
        plt.figure()
        plt.imshow(blended)
        plt.plot(polygon[:, 0], polygon[:, 1], '-', linewidth=2, c='yellow')  # Line
        # plt.show()
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.savefig(f'{segments_dir}/segment_{id:04d}.png')
        import matplotlib.path as mplPath
        polygon_path = mplPath.Path(polygon)
        # exit()
    
    cam = cfg.camera
    w, h = cam.width, cam.height
    fx, fy = cam.fx, cam.fy
    cx, cy = cam.cx, cam.cy
    dscale = cam.dscale
    N = w*h
    points = np.zeros((N, 6)) # XYZ-RGB
    i = 0
    for u in range(w): # this process is slow and cached to speedup
        for v in range(h):
            z = depth[v, u] / dscale
            points[i,0] = ((u-cx)*z)/fx # x
            points[i,1] = ((v-cy)*z)/fy # y
            points[i,2] = z             # z
            # if binary_mask[v,u] == 0:
            # if not polygon_path.contains_point((u,v)):
            if use_mask:
                if mask[v,u] == 0:
                    points[i,3:] = image[v, u]/255.0
                else:
                    points[i,3:] = np.array([1, 0, 0])
            else:
                points[i,3:] = image[v, u]/255.0
            i += 1
    
    return points

def post_process(points: np.ndarray, T):
    points = torch.tensor(points, dtype=torch.float)
    T = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180), dev='cpu')
    points[:,:3] = la.transform(points[:,:3], T)
    return points

import torch
T = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180), dev='cpu')

poses = np.load(f'./log/{dataset_name}/{scene_name}/bin/poses.npy')
print(f'Num poses : {poses.shape[0]}')


# Viusalization
vis = Vis("SIMN")
pcl = []
frunstrums = []
screenshoot_dir = f'./log/{dataset_name}/{scene_name}/screenshoot2'
# shutil.rmtree(screenshoot_dir)
os.makedirs(screenshoot_dir, exist_ok=True)
import open3d as o3d

# for i in range(poses.shape[0]):
for i in range(6):
    id = id0 + i
    Ti = torch.tensor(poses[id-id0], dtype=torch.float)
    if i < 15:
        points = post_process(generate_pc(id, scene_name), Ti@T)
        pcl.append(points)
    if i%5 == 0:
        frustrum = Vis.get_camera_frustum(cfg.camera, Ti@T, depth=3.0)
        # frunstrums.append(frustrum)
    vis.draw(pcl, None, blocking=False, 
                origin_size=0.0,
                open3d_geometries=frunstrums, 
                save_image=False) # Non-blocking
    vis.capture_screen(f'{screenshoot_dir}/screenshoot_{id:04d}.png')
    # o3d.io.write_point_cloud(f'{pcd_dir}/pcd_{self.frame_id:04d}.pcd', pcd)
    print(f'Frame {i}')


vis.destroy()