# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import pandas as pd
import numpy as np 
from lib.liealgebra import LieAlgebra as la
from torch import tensor
import torch
from lib.visualizer import Visualizer as Vis
from lib.config import Config
from rich import print
import open3d as o3d
from lib.visualizer import Visualizer as Vis 
import open3d as o3d



def view_trajectory(cfg, depth=1.0):
    if cfg.dataset.sub_path == None:
        path_file = f'./log/{cfg.dataset.name}/trajectory.csv'
    else:
        path_file = f'./log/{cfg.dataset.name}/{cfg.dataset.sub_path}/trajectory.csv' 
    # print(path_file)
    df = pd.read_csv(path_file)
    
    data = df.to_numpy()
    N = data.shape[0]
    
    translations = data[:,6:9]
    quaternions = data[:, 9:]
    dev = 'cpu'
    vis = Vis(f'View trajectory: {path_file}')
    frustrums = []
    
    axes = torch.zeros((N-1,3), device=dev)
    for i in range(N):
        translation = translations[i]
        q = quaternions[i]
        q_tensor = tensor(q).to(dev)
        rotation_matrix = la.quat_to_rot(q_tensor)
        pose = torch.eye(4).to(dev)
        pose[:3,:3] = rotation_matrix
        pose[:3,3] = tensor(translation).to(dev)
        
        if i == 0:
            T0 = pose
        else:
            Ti = pose
            p0 = la.se3_log_map(T0)
            pi = la.se3_log_map(Ti)
            rot_y = torch.abs(pi[:3] - p0[:3])
            axes[i-1,:] = rot_y
        
        # Numpy domain
        pose = pose.cpu().numpy()
        if cfg.dataset.name =='tum':
            T = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180)).to(dev).cpu().numpy()
            depth = 1.5
        elif cfg.dataset.name =='kitti':
            T = la.axis_angle_to_transform([0.0, 1.0, 0.0], la.deg2rad(90)).to(dev).cpu().numpy()
            depth = 0.3
        frustrum = Vis.get_camera_frustum(cfg.camera, pose@T, depth)
        frustrums.append(frustrum)
        
    max_vales_dim0, max_idx_dim0 = torch.max(axes, dim=0)
    # print(max_vales_dim0)
    max_val, max_idx = torch.max(max_vales_dim0, dim=-1)
    # print(f'Angle: {max_vales_dim0} -> {max_val} with axis = {max_idx}')
    vis.draw([], [], blocking=True, origin_size=0.5,  open3d_geometries=frustrums) # Non-blocking
    

def frames_to_video(cfg, dir):
    # Get a list of image files in the folder
    import cv2
    import os
    if cfg.dataset.sub_path == None:
        frame_folder = f'./log/{cfg.dataset.name}/{dir}'
        output_video = f'./log/{cfg.dataset.name}/{dir}.mp4'
    else:
        frame_folder = f'./log/{cfg.dataset.name}/{cfg.dataset.sub_path}/{dir}' 
        output_video = f'./log/{cfg.dataset.name}/{cfg.dataset.sub_path}/{dir}.mp4'

    images = sorted([img for img in os.listdir(frame_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
    if not images:
        print("No images found in the specified folder.")
        return
    # Read the first image to get dimensions
    first_image = cv2.imread(os.path.join(frame_folder, images[0]))
    height, width, _ = first_image.shape
    
    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4 files
    video = cv2.VideoWriter(output_video, fourcc, cfg.camera.fps/3.0, (width, height))
    
    # Write each frame to the video
    for image in images:
        img_path = os.path.join(frame_folder, image)
        # print('Writing file: ', img_path)
        frame = cv2.imread(img_path)
        video.write(frame)
    
    video.release()
    print(f"Video saved as {output_video}")

def show_loss(cfg):
    import os
    import matplotlib.pyplot as plt
    if cfg.dataset.sub_path == None:
        dir = f'./log/{cfg.dataset.name}'
    else:
        dir = f'./log/{cfg.dataset.name}/{cfg.dataset.sub_path}'
    lossfile = f'{dir}/loss.csv'
    figures_dir =  f'{dir}/figures'
    os.makedirs(figures_dir, exist_ok=True)
    data = pd.read_csv(lossfile).to_numpy()
    # Losses in all frames
    n = data.shape[0]
    iters = np.linspace(0, n, n)
    ncol, nrow = 1, 2
    plt.rcParams["figure.figsize"] = (6*ncol, 4*nrow)
    _, ax = plt.subplots(nrow, ncol)
    ax[0].set_title(f'Loss All Frames')
    ax[0].plot(iters, data[:, 2], label='SDF Loss')
    ax[0].plot(iters, data[:, 3], label='Eikonal Loss')
    ax[0].plot(iters, data[:, 4], label='All Loss')
    ax[0].legend()
    
    # SDF Values
    ax[1].set_title(f'SDF Values')
    ax[1].plot(iters, data[:, 5], label='SDF values')
    ax[1].legend()
    plt.savefig(f'{figures_dir}/loss_sdf_values.png')
    plt.show()
    
def view_pcd(cfg, id):
 
    
    if cfg.dataset.sub_path == None:
        base_dir = f'./log/{cfg.dataset.name}/pcd'
    else:
        base_dir = f'./log/{cfg.dataset.name}/{cfg.dataset.sub_path}/pcd' 
    path = f'{base_dir}/pcd_{id:04d}.pcd'
    print(path)
    pcd = o3d.io.read_point_cloud(path)
    Vis(f'{path}: {np.asarray(pcd.points).shape[0]} points').draw(list_pc=[], list_colors=[], blocking=True,
                        origin_size=0.0, open3d_geometries=[pcd])
    
    
def pgo(cfg):
    from scipy.spatial.distance import cdist
    
    if cfg.dataset.name =='tum':
        T = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180)).to(cfg.device).cpu().numpy()
        depth = 3.0e-3 # milimeter
    elif cfg.dataset.name =='kitti':
        T = la.axis_angle_to_transform([0.0, 1.0, 0.0], la.deg2rad(90)).to(cfg.device).cpu().numpy()
        depth = 0.3
    
    if cfg.dataset.sub_path == None:
        base_dir = f'./log/{cfg.dataset.name}/bin'
    else:
        base_dir = f'./log/{cfg.dataset.name}/{cfg.dataset.sub_path}/bin' 
        
    poses = np.load(f'{base_dir}/pose.npy')
    covs = np.load(f'{base_dir}/cov.npy')
    odoms = np.load(f'{base_dir}/odom.npy')
    pose_loop = np.load(f'{base_dir}/pose_loop.npy')
    print(poses.shape, covs.shape, odoms.shape)
    # print(pose_loop)
    
    
    frustrums = []
    N = poses.shape[0]
    vis = Vis(f'View trajectory: {base_dir}')
    loop_id, cur_id = 3, 16
    for i in range(N):
        pose = poses[i] # 1...
        if i == loop_id:
            frustrums.append(Vis.get_camera_frustum(cfg.camera, pose@T, depth, c=2))
        elif i == cur_id:
            frustrums.append(Vis.get_camera_frustum(cfg.camera, pose@T, depth, c=3))
        else:
            frustrums.append(Vis.get_camera_frustum(cfg.camera, pose@T, depth, c=1))
    
    # pose_loop
    # frustrums.append(Vis.get_camera_frustum(cfg.camera, pose_loop@T, depth, c=4))
    
    # vis.draw([], [], blocking=True, origin_size=depth,  open3d_geometries=frustrums)        
            
    # Pose Graph Optimization
    #'''
    from core.pgo import PoseGraphOptimization
    pgo = PoseGraphOptimization()
    pgo.add_prior(0, poses[0])
    pgo.add_init(0, init_pose=np.eye(4))
    
    for i in range (N-1):
        pgo.add_odom(i, i+1, odoms[i], covs[i])
        pgo.add_init(i+1, poses[i+1])
        
        
    sdist = 0.0
    for i in range(N-1):
        pos = odoms[i, :3, 3]
        # print(pos)
        sdist += np.sum(pos**2, axis=-1)**0.5
        
    print(f'sdist1 = {sdist}')
    
    ## Add loop
    odom = np.linalg.inv(poses[loop_id])@poses[cur_id]
    pgo.add_odom(loop_id, cur_id, odom)
    
    sdist = 0.0
    est_poses = pgo.optimize()
    for i, pose in  enumerate(est_poses):
        pose[:3,3] = pose[:3,3] + np.array([0.0, 0.0, 5.0e-2])
        if i == loop_id:
            frustrums.append(Vis.get_camera_frustum(cfg.camera, pose@T, depth, c=2))
        elif i == cur_id:
            frustrums.append(Vis.get_camera_frustum(cfg.camera, pose@T, depth, c=3))
        else:
            frustrums.append(Vis.get_camera_frustum(cfg.camera, pose@T, depth, c=1))
            
        if i > 0 :
            odom = np.linalg.inv(prev_pose)@pose
            pos = odom[:3, 3]
            sdist += np.sum(pos**2, axis=-1)**0.5
            
        prev_pose = pose
    print(f'sdist2 = {sdist}')
    #'''
        
    # vis.draw([], [], blocking=True, origin_size=depth,  open3d_geometries=frustrums)
    
# Examples:
# python tools.py --cmd loss -c config/tum.yaml
# python tools.py --cmd traj -c config/kitti.yaml
# python tools.py --cmd pcd -c config/tum.yaml -i 180
# python tools.py --cmd video -c config/uta.yaml -d screenshoot
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIMN")
    parser.add_argument('--cmd', '-cmd', type=str, default='traj')
    parser.add_argument('--cfg', '-c', type=str, default='config/tum.yaml')
    parser.add_argument('--id', '-i', type=int, default=0)
    parser.add_argument('--dir', '-d', type=str, default='screenshoot')
   
    args = parser.parse_args()

    # Read configuration from dataset
    cfg = Config(path=f'{args.cfg}').config()

    if args.cmd == 'traj':
        view_trajectory(cfg)
    elif args.cmd == 'video':
        frames_to_video(cfg, args.dir)
    elif args.cmd == 'loss':
        show_loss(cfg)
    elif args.cmd == 'pcd':
        view_pcd(cfg, args.id)
    elif args.cmd == 'pgo':
        pgo(cfg)
    else:
        print('No command found!')