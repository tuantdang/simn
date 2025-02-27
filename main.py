# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

from dataset.kittidataset import KittiDataset
from core.registration import Registration
from core.sampling import Sampling
import torch
from lib.liealgebra import LieAlgebra as la
from lib.visualizer import Visualizer as Vis
from core.latentfeature import LatentFeature
from core.training import TrainingPool
from rich import print
from lib.config import Config
from core.models import Decoder, PCDTransformer
from lib.logger import Logger
from torch import nn
from dataset.tumdataset import TumDataset
from dataset.utadataset import UtaDataset
from dataset.realsense_cam import RealsenseCam
import open3d as o3d
from lib.utils import print_var
import time
import numpy as np
import os
import signal

running = True

def signal_handler(signum, frame):
    global running
    running = False
    print(f'\nSIGINT received: signum = {signum}, frame={frame}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SIMN")
    parser.add_argument('--cfg', '-c', type=str, default='config/tum.yaml')
    parser.add_argument('--toframeid', '-i', type=int, default=-1)
    
    signal.signal(signal.SIGINT, signal_handler)
    args = parser.parse_args()
    cfg = Config(path=f'{args.cfg}').config()
    
    torch.manual_seed(cfg.sampling.seed) # Seed
    logger = Logger(cfg, print_on=True)
    
    # Dataset & sampler
    if cfg.dataset.name == 'kitti':
        dataset = KittiDataset(cfg, logger)
    elif cfg.dataset.name == 'tum':
        dataset = TumDataset(cfg, logger)
    elif cfg.dataset.name == 'uta':
        dataset = UtaDataset(cfg, logger)
    elif cfg.dataset.name == 'realsene':
        dataset = RealsenseCam(cfg, logger)
        
    sampler = Sampling(cfg, logger)
    
    # Point Features, TrainingPool, and Registrationing
    decoder = Decoder(cfg); print(decoder)
    # decoder = PCDTransformer(cfg); print(decoder)
    def custom_init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)
    decoder.apply(custom_init)

    latentfeature = LatentFeature(cfg, logger)
    trainner = TrainingPool(cfg, latentfeature, decoder, logger)
    reg = Registration(cfg, latentfeature, decoder, logger)
    
    vis = Vis("SIMN", logger)
    
    if args.toframeid < 0:
        exit_frame_id = dataset.pc_count-1 # all frames
    else:
        exit_frame_id = args.toframeid # to the frame id then stop
    
    if cfg.dataset.name == 'realsene':
        dataset.skip_first_frames()

    poses = []    
    while dataset.available():
        
        frame_id, pc, _, _ = dataset.next_frame()
            
        if frame_id == 0:
            frame_pose = torch.eye(4, device=pc.device)
        else:  # Perform Registrationing Here
            frame_pose = reg.register(pc, frame_id) # Perform registration with new data
            reg.detect_new_local_frame()
        
        # samples = Sampling.plane_sample(points, K=3, surface_range=0.1)
        # samples, displacements, weights = sampler.plane_radius_sample(pc)
        samples, displacements, weights = sampler.dist_sampling(pc)
        # samples, displacements, weights = sampler.sample(points)
        
        
        if cfg.visual_debug:
        # if True:
            Vis(f"Up Samples at FrameId: {frame_id:03d}").draw([samples], [Vis.red])
        
        latentfeature.update(frame_id, samples, displacements, frame_pose) # point feature update!
        trainner.update(frame_id, frame_pose, samples, displacements, weights) # update trainner with upsamples
        
        # Log trajectory
        Nf, Nm = latentfeature.points.shape[0], trainner.points.shape[0]
        txt = f'{frame_id:03d}, {Nf:010d}, {Nm:010d}, {Nm/Nf:03.1f}, '
        txt += la.view_transform(frame_pose)
        logger.write_traj(txt)
        poses.append(frame_pose.cpu().numpy())

        # Log est
        est_txt = str(dataset.ts[frame_id]) + " "
        est_txt += la.view_transform2(frame_pose)
        logger.write_est(est_txt)
        
        # Draw continous frames
        if cfg.visual:
            # print_var(latentfeature.points, 'latentfeature.points')
            if cfg.dataset.name == 'kitti':
                vis.draw([latentfeature.points], [Vis.red], blocking=False, 
                         origin_size=3.0, save_image=True) # Non-blocking
            elif cfg.dataset.name == 'tum' or cfg.dataset.name == 'realsene' or cfg.dataset.name == 'uta':
                camera = cfg.camera
                T1 = frame_pose.cpu().numpy()
                T2 = dataset.Tx180Neg.cpu().numpy()
                frustrum = Vis.get_camera_frustum(camera, T1@T2, depth=3.0)
                
                view_points = latentfeature.points.clone()
                view_points[:, [3,5]] = view_points[:, [5,3]]
                vis.draw([view_points], None, blocking=False, origin_size=0.0,
                         open3d_geometries=[frustrum],
                         save_image=True,
                         save_pcd_rate=cfg.save_pcd_rate) # Non-blocking
        
        if frame_id >= exit_frame_id or not running:  # condition to exit for debugging    
            print('Saving and Cleanup....')
            # np.savetxt(f'{logger.dir}/covariances.csv', reg.cov_poses.cpu().numpy().reshape((-1,6)), fmt="%+0.5f", delimiter=",")
            bin_dir = f'{logger.dir}/bin'
            os.makedirs(bin_dir, exist_ok=True)
            np.save(f'{bin_dir}/cov.npy', reg.cov_poses.cpu().numpy())
            np.save(f'{bin_dir}/pose.npy', reg.poses.cpu().numpy())
            np.save(f'{bin_dir}/odom.npy', reg.odometry.cpu().numpy())
            np.save(f'{bin_dir}/poses.npy', np.asanyarray(poses))
            # np.savetxt(f'{bin_dir}/local_frame_id.txt', np.array(reg.local_frame_id), fmt="%03d", delimiter=",")
            # break
            
            logger.close()
            logger.generate_loss_charts(frame_id+1)
            vis.destroy()
            exit()
        # End of loop    
    # Return Main    
   
    