# @author    Tuan Dang   
# Email: tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import torch
from torch import Tensor, optim
from rich import print
from core.latentfeature import LatentFeature
import math
from lib.utils import print_var
from core.loss import *
from core.models import Decoder
from lib.logger import Logger
from lib.liealgebra import LieAlgebra as la

class TrainingPool:
    def __init__(self, cfg, latentfeature:LatentFeature=None, decoder:Decoder=None, logger:Logger=None):
        self.cfg = cfg
        self.points = torch.empty((0,3), device=cfg.device, dtype=cfg.dtype.point) # [0,3]: point in T0 frame
        self.latentfeature = latentfeature
        self.decoder = decoder # shared decoder
        self.logger = logger
        
        # TrainingPool data 
        self.points = torch.empty((0, cfg.feature.input_dim), device=cfg.device, dtype=cfg.dtype.point) # [N,F] point in T0 frame
        self.sdf_values = torch.empty((0), device=cfg.device, dtype=cfg.dtype.point) # [N]
        self.weights = torch.empty((0), device=cfg.device, dtype=cfg.dtype.point) # [N] point weight higher near surface 
        self.timestamp = torch.empty((0), device=cfg.device, dtype=cfg.dtype.index) # [N]
        self.frame_id = 0
        
        self.pc_list = []
    
    '''
    - Add new training samples into pool
    - Find the close to the surface samples from input samples
    - Train the decoder with the training samples
    '''    
    def update(self,    frame_id,
                        pose:Tensor, # frame pose, tensor([4,4]),
                        samples:Tensor,  #[N,3]: update from upsampling
                        sdf_values:Tensor,  #[N]
                        weights:Tensor,  #[N]
                ):
        self.frame_id = frame_id
        N0 = self.points.shape[0] # Number of points before updating
        Ns = samples.shape[0] # Number of samples
        frame_origin = pose[0:3,3] # [3,1] Current frame T position
        samples_T0 = samples.clone()
        samples_T0[:,:3] = la.transform(samples[:,:3], pose) # [N, F] samples in T0
        
        # Add to samples to training data pool
        self.points = torch.cat([self.points, samples_T0], dim=0) # points in T0
        self.sdf_values = torch.cat([self.sdf_values, sdf_values], dim=0)
        self.weights = torch.cat([self.weights, weights], dim=0)
        new_ts = torch.ones(Ns, device=self.cfg.device, dtype=self.cfg.dtype.index)*frame_id
        self.timestamp = torch.cat([self.timestamp, new_ts], dim=0)
        
        N1 = self.points.shape[0] # Number of point after concatenating with samples
        
        points_in_cur_frame = self.points[:,:3] - frame_origin # [N1,3] move all points current frame T
        dist = torch.norm(points_in_cur_frame, p=2, dim=-1) # [N1]  point distances to origin of frame T
        
        # Filter out points out of frame radius
        mask = dist < self.cfg.local_frame.radius # [N1]
        N2 = mask.sum() # number of valid points with radius filter
        if N2 > self.cfg.local_frame.max_points: # overflowed frame
            Nd = N2 - self.cfg.local_frame.max_points # Number of discarded points
            indices = torch.randperm(N2)[:Nd] # choose random [Nd] indices out of N2
            mask[indices] = False # Mask discarded points out of  filtered_points
            self.logger.log(1, f'  TrainingPool->update:  All points now is {N2} > {self.cfg.local_frame.max_points} frame buffer: discard {Nd} points')
        
        # Update training with filter mask: distance and overflow filters    
        self.points = self.points[mask]
        self.sdf_values = self.sdf_values[mask]
        self.weights = self.weights[mask]
        self.timestamp = self.timestamp[mask]
        
        #================ Processing TrainingPool Data =============================
        N3 = self.points.shape[0] # number of point with overlow filter
        Nvs = mask[-Ns:].sum() # Number of valid samples: take last Ns mask, then count the valid ones
        valid_samples = self.points[-Nvs:] # valid samples: most recent ones
        # self.logger.log(1, f'  TrainingPool->update: samples Ns={Ns}, points: before={N0}, after={N1}, --radius-filter--> N2={N2} -overflow-> N3={N3}')
        
        
        
        # Find sample indices close to the surface
        # Query points cerntainties
        bs = self. cfg.batchsize.infer
        n_iter = math.ceil(Nvs/bs)
        
        sampled_sdf_values = self.sdf_values[-Nvs:]
        sampled_certainties = torch.zeros((Nvs)).to(valid_samples)
        for iter in range(n_iter):
            head = iter*bs
            tail = min((iter+1)*bs, Nvs)
            batch_samples = valid_samples[head:tail, :]
            batch_certainties = self.latentfeature.query_certainty(batch_samples[:,:3])
            sampled_certainties[head:tail] = batch_certainties # frame_id=0 -> [378510] points
            # self.logger.log(1, f'  TrainingPool->update: batch_samples={batch_samples.shape}, batch_certainties={batch_certainties.shape} -> {batch_certainties[0:15]}')
        
        # self.logger.log(1, f'  TrainingPool->update: Ns={Ns}, Nvs={Nvs} (valid samples), bs={bs}, n_iter={n_iter}, sampled_certainties_mean={sampled_certainties.mean():0.4f}')
        close_surface_sample_idx = torch.where((sampled_certainties < self.cfg.feature.certainty_thres) # valid certainty
                                 & (torch.abs(sampled_sdf_values) < self.cfg.sampling.surface_sample_range*3.0) # close to surface
                                 )[0]
        self.logger.log(1, f'  TrainingPool->update: Nvs={Nvs} ->  close_surface_sample_idx={close_surface_sample_idx.shape[0]}, all_points={self.points.shape[0]}, increasement: {self.points.shape[0] -Nvs}')
        close_surface_sample_idx += (self.points.shape[0] - Nvs)
        
        self.train(close_surface_sample_idx, self.cfg.training.n_iter) # Train with close to the surface samples
        
        ''' Visualization: samples, valid samples, close to surface samples
        from visualizer import Visualizer
        close_surface_samples = self.points[close_surface_sample_idx]
        red, green, blue, black = [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]
        vis_pcs = Visualizer.get_pointclouds([samples, valid_samples, close_surface_samples], [green, blue, red],  origin_size=2)
        Visualizer.draw_geometries(vis_pcs)
        '''
        
        ''' Visualize added samples to pool overtime
        self.pc_list.append(valid_samples)
        npc = len(self.pc_list)
        if frame_id == 10:
            from visualizer import Visualizer
            colors, names = Visualizer.get_colors()
            vis_pcs = Visualizer.get_pointclouds(self.pc_list, colors[0:npc],  origin_size=2)
            Visualizer.draw_geometries(vis_pcs)
        '''
    def train(self, surface_sample_idx:Tensor, n_iter=480):
        
        step = self.cfg.batchsize.skip_step
        latentfeature_param = list(self.latentfeature.parameters())
        decoder_param =  list(self.decoder.parameters())
        optimizer = self.setup_optim(latentfeature_param, decoder_param)
        self.logger.log(1, f'  TrainingPool->train: n_iter={n_iter}, surface_sample_idx={surface_sample_idx.shape}')
        
        # Adaptive iteration training
        losses = torch.empty((0), device=self.cfg.device, dtype=self.cfg.dtype.point)
        window_size = self.cfg.training.window
        mean_loss = 0.0
        for iter in range(n_iter):
            new_sample_ratio = self.cfg.batchsize.new_sample_ratio
            bs = self.cfg.batchsize.training
            (batch_samples, batch_sdf_values, #[bs, 3], [bs]
             batch_weights, batch_timestamp # [bs], [bs]
            ) = self.get_training_samples(surface_sample_idx, new_sample_ratio, bs)
            
            (feature_vector, # [bs, F] 
             weight_vector, _, #  [bs, knn] 
            ) = self.latentfeature.query_features(batch_samples, training=True)
            batch_pred_sdf = self.decoder.sdf(feature_vector) # [bs]
            
            # Loss
            wb, we = self.cfg.loss.weight_bce, self.cfg.loss.weight_eikonal
            sdf_loss = sdf_bce_loss(batch_pred_sdf, batch_sdf_values, weights=None)
            grad = self.get_grad_xyz(batch_samples[::step]) # [bs/10, 3] downsampleing by factor step
            eikonal_loss = ((grad.norm(p=2, dim=-1) - 1)**2).mean() # [bs/step, 3] -> [bs/step]: mean square error for length of gradient
            loss = wb*sdf_loss + we*eikonal_loss
            
            
            # Only for debugging to see backprobagation to feature domain for each iteration
            if self.frame_id == 0:
                m_feature = self.latentfeature.learned_features.data.mean(dim=0).cpu().numpy() # [F]
                txt = ""
                for i, v in enumerate(m_feature):
                    if i != m_feature.shape[0] - 1:
                        txt += f'{v:+0.4f}, '
                    else: # last item
                        txt += f'{v:+0.4f}'
                self.logger.write_learned_feature(txt)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=False)
            optimizer.step()
            
            self.logger.write_loss(f'{self.frame_id:03d}, {iter:03d}, {wb*sdf_loss:0.3f}, {we*eikonal_loss:0.3f}, {loss:0.3f}, {batch_sdf_values.mean():0.3f}')
            
            # Apdative iteration
            if self.cfg.training.adaptive:
                losses = torch.cat([losses, loss.view(-1,1)], dim=0)
                losses_count = losses.shape[0]
                thres = self.cfg.training.loss_threshold
                if losses_count >= window_size:
                    cur_mean_loss = losses[-window_size:].mean().item() # mean last window_size items
                    if (abs(cur_mean_loss - mean_loss) < thres):
                        self.logger.log(1, f'  TrainingPool->train: break at iter={iter}, cur_mean_loss={cur_mean_loss:0.6f}, mean_loss={mean_loss:0.6f}, diff={cur_mean_loss-mean_loss:0.6f}, thres={thres:0.6f}')
                        break
                    else:
                        if losses_count % int(window_size): # not update mean_loss every iter
                            mean_loss = cur_mean_loss
                
            
        
    def get_training_samples(self, surface_sample_idx:Tensor, 
                             new_sample_ratio=0.125, 
                             bs=16384):
        '''  Gather training input with 2 parts into batch configured at cfg.batchsize.training
        1. New samples from sample closes to the surface
        2. Old sample in the pool: points
        '''
        N = self.points.shape[0] # all points in pool
        Ns =  surface_sample_idx.shape[0] # number of sampled points
        new_sample_count = math.ceil(new_sample_ratio*bs) # close to the surface sample count (~2K)
        new_sample_count = min(new_sample_count, Ns) # in case less surface samples than cofiguration
        bs = min(bs, N) # in ase all number of points are less than batch size
        old_sample_count = bs - new_sample_count # old sample count in the pool (~14K)
        if old_sample_count < 0:
            exit(f'  TrainingPool->get_training_samples: old_sample_count={old_sample_count}. Not enough point to train')
       
        old_sample_idx = torch.randint(0, N, (old_sample_count,), device=self.cfg.device) # [old_sample_count] choose old_sample_count out of N
        new_sample_relative_idx = torch.randint(0, Ns, (new_sample_count,), device=self.cfg.device) # choose new_sample_count out of Ns
        new_sample_idx = surface_sample_idx[new_sample_relative_idx] # [new_sample_idx] translate relative sample idx to pool idx
        idx = torch.cat([old_sample_idx, new_sample_idx], dim=0) # [bs]
        
        batch_samples = self.points[idx] # batch_samples~16k, self.points~500k (frame_id=0)
        batch_sdf_values = self.sdf_values[idx]
        batch_weights = self.weights[idx]
        batch_timestamp = self.timestamp[idx]
        
        
        ''' Visualization # close to the surface points vs pool
        from visualizer import Visualizer
        red, green, blue, black = [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]
        vis_pcs = Visualizer.get_pointclouds([self.points[old_sample_idx], self.points[new_sample_idx]], [green, red],  origin_size=2)
        Visualizer.draw_geometries(vis_pcs)
        '''
        return batch_samples, batch_sdf_values, batch_weights, batch_timestamp
    
    '''
     Referen from PIN-SLAM
    '''   
    def get_grad_xyz(self,
                 samples:Tensor, # [N,3]
                 delta = 0.06
                 ):
        xyz = samples[:,:3]
        delta_x = torch.tensor([delta, 0.0, 0.0], device=self.cfg.device, dtype=self.cfg.dtype.point) # [1,3]
        delta_y = torch.tensor([0.0, delta, 0.0], device=self.cfg.device, dtype=self.cfg.dtype.point) # [1,3]
        delta_z = torch.tensor([0.0, 0.0, delta], device=self.cfg.device, dtype=self.cfg.dtype.point) # [1,3]
        
        x_pos, x_neg = xyz + delta_x, xyz - delta_x # [N,3] +/- [1,3] = [N,3]
        y_pos, y_neg = xyz + delta_y, xyz - delta_y # [N,3] +/- [1,3] = [N,3]
        z_pos, z_neg = xyz + delta_z, xyz - delta_z # [N,3] +/- [1,3] = [N,3]
        
        samples_x6 = torch.cat([x_pos, x_neg, y_pos, y_neg, z_pos, z_neg], dim=0) # [N*6, 3]
        feature_x6, _, _ = self.latentfeature.query_features(samples_x6, training=False)
        sdf_x6 = self.decoder.sdf(feature_x6) # [N*6]
        
        N = xyz.shape[0]
        sdf_x_pos, sdf_x_neg = sdf_x6[:N], sdf_x6[N:N*2]
        sdf_y_pos, sdf_y_neg = sdf_x6[N*2:N*3], sdf_x6[N*3:N*4]
        sdf_z_pos, sdf_z_neg = sdf_x6[N*4:N*5], sdf_x6[N*5:N*6]
        
        grad_x = (sdf_x_pos - sdf_x_neg)/(2*delta) # [N]
        grad_y = (sdf_y_pos - sdf_y_neg)/(2*delta) # [N]
        grad_z = (sdf_z_pos - sdf_z_neg)/(2*delta) # [N]
        
        grad = torch.cat([grad_x.unsqueeze(-1), grad_y.unsqueeze(-1), grad_z.unsqueeze(-1)], dim=-1) # [N,3]    
        return grad
    
    def setup_optim(self, latentfeature_param, decoder_param):
        latentfeature_opt = {
            "params": latentfeature_param,
            "lr": self.cfg.optimizer.learning_rate,
            "weight_decay": self.cfg.optimizer.weight_decay,
        }
        
        decoder_opt = {
            "params": decoder_param,
            "lr": self.cfg.optimizer.learning_rate,
            "weight_decay": self.cfg.optimizer.weight_decay,
        }
        
        opt_settings = [latentfeature_opt, decoder_opt]
        
        if self.cfg.optimizer.name == "Adam":
            opt = optim.Adam(opt_settings, 
                             betas=(0.9, 0.99), 
                             eps=self.cfg.optimizer.esp)
        else:
            opt = optim.SGD(opt_settings, momentum=0.9)
        
        return opt