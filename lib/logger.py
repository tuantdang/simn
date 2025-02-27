# @author    Tuan Dang   
# Email: tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved


import logging
# from os.path import exists, isdir
import os
from rich import print
from datetime import datetime
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class Logger:
     def __init__(self, cfg, print_on=True):
          self.cfg = cfg
          self.print_on = print_on
          
          if cfg.dataset.sub_path == None:
               dir = f'./log/{cfg.dataset.name}'
          else:
               dir = f'./log/{cfg.dataset.name}/{cfg.dataset.sub_path}'
          self.dir = dir
          
          os.makedirs(dir, exist_ok=True)
          self.files = []
          
          self.logfile = open( f'{dir}/log.txt', 'w')
          self.logfile.write("Start logging at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
          self.files.append(self.logfile)
          
          self.lossfile = open( f'{dir}/loss.csv', 'w')
          self.lossfile.write(f'Id, Iter, SDF, Eikonal, Loss, SDF Values \n')
          self.files.append(self.lossfile)
          
          self.weightfile = open( f'{dir}/weight.csv', 'w')
          self.weightfile.write(f'FID, Iter, Num1, Num2,  w_grad,  w_err \n')
          self.files.append(self.weightfile)
          
          self.sdffile = open( f'{dir}/sdf.csv', 'w')
          self.sdffile.write(f'Num1, Num2, muSDF, gradLen,  gradX,  gradY,   gradZ \n')
          self.files.append(self.sdffile)
          
          self.posefile = open( f'{dir}/pose.csv', 'w')
          self.posefile.write(f'FID, Iter, Num1, Num2,  dAngle,    Trans,   dTx,    dTy,   dTz,     dqx,    dqy,    dqz,   dqw  \n')
          self.files.append(self.posefile)
                    
          self.trajfile = open( f'{dir}/trajectory.csv', 'w')
          self.trajfile.write(f'FID,    NpFeat,      NpMap,  M/F,   Angle,    Trans,    Tx,      Ty,    Tz,      qx,     qy,     qz,     qw  \n')
          self.files.append(self.trajfile)

          self.estfile = open( f'{dir}/est.csv', 'w')
          self.estfile.write(f'#timestamp tx ty tz qx qy qz qw \n')
          self.files.append(self.estfile)
          
          self.learned_feature_file = open( f'{dir}/learned_feature.csv', 'w')
          self.files.append(self.learned_feature_file)
          
          self.track_feature_file = open( f'{dir}/track_feature.csv', 'w') # View features when tracking
          self.files.append(self.track_feature_file)
          
          self.cov_file = open( f'{dir}/cov.csv', 'w') 
          self.files.append(self.cov_file)
          
          self.latent_sampling_file = open( f'{dir}/latent_sampling.csv', 'w')
          self.latent_sampling_file.write(f'Samples,    CloseSurfa, VxDwSamlin, KnnFilter,  NewPoints,  AllPoints \n')
          self.files.append(self.latent_sampling_file)
          
          self.local_frameid_file = open( f'{dir}/local_frames.csv', 'w')
          self.local_frameid_file.write(f'FrmId, RotX,  RotY,  RotZ,  AccTrans, nFrames\n')
          self.files.append(self.local_frameid_file)
          
          self.loop_file = open( f'{dir}/loop.csv', 'w')
          self.loop_file.write(f'CurId, StrId, LpId, MinDist\n')
          self.files.append(self.loop_file)
          
          self.range_file = open( f'{dir}/range.csv', 'w')
          self.range_file.write(f'FrmId,   MinX,    MaxX,    MinY,    MaxY,    MinZ,    MaxZ\n')
          self.files.append(self.range_file)
                
        
     def log(self, verbose_thres:int=1, str=None):
        if self.cfg.verbose >= verbose_thres:
            if self.print_on:
                print(str)
            self.logfile.write(str+"\n")
            
     def close(self): # close all files
        for f in self.files:
            f.close()
            
     def write_loss(self, str):
         self.lossfile.write(str+"\n")
         
     def write_weight(self, str):
         self.weightfile.write(str+"\n")
         
     def write_sdf(self, str):
         self.sdffile.write(str+"\n")
         
     def write_pose(self, str):
         self.posefile.write(str+"\n")
         
     def write_traj(self, str):
         self.trajfile.write(str+"\n")
    
     def write_est(self, str):
         self.estfile.write(str+"\n")
         
     def write_learned_feature(self, str):
         self.learned_feature_file.write(str+"\n")
         
     def write_track_feature(self, str):
         self.track_feature_file.write(str+"\n")
         
     def write_cov(self, str):
         self.cov_file.write(str+"\n")
         
     def write_latent_samling(self, str):
         self.latent_sampling_file.write(str+"\n")
         
     def write_local_frameid(self, str):
         self.local_frameid_file.write(str+"\n")
         
     def write_loop(self, str):
         self.loop_file.write(str+"\n")
         
     def write_range(self, str):
         self.range_file.write(str+"\n")
         
     def generate_loss_charts(self, n_frames=2, loss_each_frame=False, show=False):
     
        lossfile = f'{self.dir}/loss.csv'
        figures_dir =  f'{self.dir}/figures'
        os.makedirs(figures_dir, exist_ok=True)
        data = pd.read_csv(lossfile).to_numpy()
        
        # Losses in each frame
        if loss_each_frame:
            n_iter = self.cfg.n_iter_training
            for frame_id in range(n_frames):
                _, ax = plt.subplots(1, 1)
                data_frame_id = data[frame_id*n_iter:(frame_id+1)*n_iter,]
                iters = data_frame_id[:, 1]
                sdf_loss = data_frame_id[:, 2]
                eik_loss = data_frame_id[:, 3]
                all_loss = data_frame_id[:, 4]
                
                ax.set_title(f'Loss Frame Id {frame_id:03d}')
                ax.plot(iters, sdf_loss, label='SDF Loss')
                ax.plot(iters, eik_loss, label='Eikonal Loss')
                ax.plot(iters, all_loss, label='All Loss')
                ax.legend()
                plt.savefig(f'{figures_dir}/{frame_id:03d}.png')
        
        # Losses in all frames
        n = data.shape[0]
        iters = np.linspace(0, n, n)
        _, ax = plt.subplots(1, 1)
        ax.set_title(f'Loss All Frames')
        ax.plot(iters, data[:, 2], label='SDF Loss')
        ax.plot(iters, data[:, 3], label='Eikonal Loss')
        ax.plot(iters, data[:, 4], label='All Loss')
        ax.legend()
        plt.savefig(f'{figures_dir}/all_losses.png')
        
        # SDF Values
        _, ax = plt.subplots(1, 1)
        ax.set_title(f'SDF Values')
        ax.plot(iters, data[:, 5], label='SDF values')
        ax.legend()
        plt.savefig(f'{figures_dir}/sdf_values.png')
        if show:
            plt.show()