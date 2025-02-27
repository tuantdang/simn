# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2024 Tuan Dang, all rights reserved

from torch import tensor, Tensor, nn
import torch
from rich import print # print with color format
from lib.liealgebra import LieAlgebra as la
import math
from lib.utils import print_var, hash, point2grid, create_neighbors_grid, voxel_down_sample
import numpy as np
from lib.logger import Logger
from lib.visualizer import Visualizer as Vis
import os

class LatentFeature(nn.Module):
    def __init__(self, cfg=None, logger:Logger=None):
        super().__init__()
         
        self.cfg = cfg
        self.logger = logger
        self.buffer_size = cfg.buffer_size # The maxium number of points to be processed
        self.resolution = cfg.pc.vs_latentfeature # Grid resolution
        
        self.input_dim = cfg.feature.input_dim
        self.points = torch.empty((0, self.input_dim), device=cfg.device, dtype=cfg.dtype.point) #[0,1,2]: X,Y,Z and more
        self.timestamp = torch.empty((0), device=cfg.device, dtype=cfg.dtype.index)
        self.features = torch.empty((0, cfg.feature.feature_dim), device=cfg.device, dtype=cfg.dtype.point)
        self.certainties = torch.empty((0), device=cfg.device, dtype=cfg.dtype.point)
        
        # Point indices 
        self.point_index = torch.full((self.buffer_size,), -1, 
                                          dtype=cfg.dtype.index, device=cfg.device) # Filled with -1
        self.travel_dist = torch.zeros(cfg.max_number_frames, dtype=cfg.dtype.point, device=cfg.device)
        self.frame_id = 0
        
        # Learned features
        self.learned_features = nn.Parameter()
        
        # Debug information
        self.print_tracking = False
        self.mode = 0 # 1:train, 2:track
        self.pc_list = []
        self.ts_npoints = []
        # self.ts_npoints.append(0)
        
    def is_empty(self):
        return self.points.shape[0] == 0
    
    '''
    3D points -> samples (voxel size)  -> 3D grid coordinates (voxel size) -> hash to single index using custom polynomial hash
    
    '''
    def update(self, frame_id:int, # current frame id
               samples: Tensor, # [N0,3]
               displacements: Tensor, # [N]
               pose: Tensor
               ):
        self.frame_id = frame_id
        N0 = samples.shape[0]
        sdf_threshold = self.cfg.sampling.sdf_threshold
        samples = samples[torch.abs(displacements) < sdf_threshold] # close-to-surface
        Ncf = samples.shape[0]
        samples_T0 = samples.clone()
        samples_T0[:,:3] = la.transform(samples[:,:3], pose)  # To T0 frame
        # Downsample for updating points into map
        _ , idx = voxel_down_sample(samples_T0, self.resolution, self.cfg.verbose) # [Nvd,3]
        
        sampled_points = samples_T0[idx]
        Nvd = sampled_points.shape[0] # Voxel Down Sampling
        idx = self.knn_filter(sampled_points)
        sampled_points = sampled_points[idx]
        Nknn = sampled_points.shape[0] # After nkk filter
        
        # Consider applying another filter for points overlap (same voxel) with existing points (depends on Registration)
        
        # Consider re-downsampling after sometimes
        
        # Convert world coordinate [X,Y,Z] to single index in point_index using hash function and lookup array 
        sample_grid = point2grid(sampled_points[:,:3], self.resolution) # [Nknn,3]
        sample_hash = hash(sample_grid, self.buffer_size) # [Nknn]
        sample_idx = self.point_index[sample_hash] # [Nknn] lookup in point index
        
        if not self.is_empty():
            sample_points_in_local_frame = self.points[sample_idx] #[Nknn, 3]
            vec = sample_points_in_local_frame[:,:3] - sampled_points[0,:3] # [Nknn, 3]
            dist = torch.norm(vec, p=2, dim=-1)# [Nknn]
            mask = (sample_idx == -1) | (dist > math.sqrt(3)*self.resolution) # [Nknn]: neighbor_idx == -1 new point
            
             # Accumulated travel distance
            translation_vec = pose[:3,3]
            translation = torch.norm(translation_vec, p=2) # sum(x^p, dim=-1)^(1/p)
            self.travel_dist[frame_id] = self.travel_dist[frame_id - 1] + translation # update accumulated travel distance
           
            cur_points_ts = self.timestamp[sample_idx] # Nknn
            cur_acc_dist = self.travel_dist[frame_id] # scalar [1]
            diff_travel_dist = cur_acc_dist  - self.travel_dist[cur_points_ts] # [1] - [Nknn] = [Nknn]
            diff_travel_dist_threshold = self.cfg.local_frame.radius*self.cfg.local_frame.travel_dist_ratio # 62*5=310
            mask = mask | (diff_travel_dist > diff_travel_dist_threshold) # [Nknn]
            self.logger.log(2, f'  LatentFeature->update: points NOT EMPTY')
            self.logger.log(2, f'    > Number of duplicated points in previous points:  {(sample_idx != -1).sum()}/{sample_idx.shape[0]}')
            self.logger.log(2, f'    > Add NOT duplicated points with distance greater than resolution:   {(mask > 0).sum()}/{mask.shape[0]}')
            self.logger.log(2, f'    > translation_vec={translation_vec}, translation={translation}')
            self.logger.log(2, f'    > Add points travel far away:                                        {(mask > 0).sum()}/{mask.shape[0]}')
        else:
            self.logger.log(2, f'  LatentFeature->update: points are EMPTY. No duplication')
            mask = torch.ones(Nknn, dtype=torch.bool, device=self.cfg.device) # [Nknn] 
            
        new_points = sampled_points[mask] # [Nnew] where Nnew <= Nknn <= N0
        Nnew = new_points.shape[0]
        
        
        feature_dim = self.cfg.feature.feature_dim # F
        feature_std = self.cfg.feature.feature_std # 0.0
        # feature_std = 0.1
        # if self.frame_id == 0:
        if True:
            new_features = torch.randn(Nnew, feature_dim, dtype=self.cfg.dtype.point, device=self.cfg.device)*feature_std
        else: # Interpolate from existing neural points
            feature_vector, _, _ = self.query_features(new_points, training=False) # [N,F+input_dim]
            new_features = feature_vector[:,:feature_dim].clone()
        
        self.features = torch.cat([self.features, new_features], dim=0) # [N+Nnew, feature_dim]
        self.learned_features = nn.Parameter(self.features)
        
        # New certainties
        new_certainty = torch.zeros((Nnew), dtype=self.cfg.dtype.point, device=self.cfg.device)
        self.certainties = torch.cat([self.certainties, new_certainty], dim=0) # [N+Nnew]
        
       
       # Update points index using hash
        N = self.points.shape[0] # current total points
        point_indices = torch.zeros(Nknn, dtype=self.cfg.dtype.index, device=self.cfg.device) #[Nknn]
        point_indices[mask] = torch.arange(Nnew,  dtype=self.cfg.dtype.index, device=self.cfg.device) + N # indices: N..(N+Nnew-1)
        self.point_index[sample_hash] = point_indices
        
        
        # Add new points to point manager
        self.points = torch.cat([self.points, new_points], dim=0) # [N+Nnew, F]
        Nall = self.points.shape[0]
        self.logger.log(1, f'  LatentFeature->update: samples={N0}, new_points={Nnew} (closreSur>VoxDwn>Radius>Distance), all_points={Nall}')
        self.logger.write_latent_samling(f'{N0:010d}, {Ncf:010d}, {Nvd:010d}, {Nknn:010d}, {Nnew:010d}, {Nall:010d}')
        
        # last_index = self.ts_npoints[-1]
        self.ts_npoints.append(self.points.shape[0])
        # print(f'  LatentFeature->update: {self.ts_npoints}')        
        # current timestamp and concate to new points
        new_ts = torch.ones((Nnew),  dtype=self.cfg.dtype.index, device=self.cfg.device)*frame_id 
        self.timestamp = torch.cat([self.timestamp, new_ts], dim=0) # [N+Nnew]
        
      
        
        #''' Visualize added samples to map overtime with different colors
        if self.cfg.visual_debug and self.cfg.dataset.name == 'kitti':
        # if True:
            self.pc_list.append(new_points)
            npc = len(self.pc_list)
            if frame_id == 24:
                if self.cfg.dataset.name == 'kitti':
                    Vis("LatentFeature: Update Multi-pcs->kitti").draw(self.pc_list, Vis.colors[:npc], blocking=True, origin_size=3.0)
                elif self.cfg.dataset.name == 'tum':
                    Vis("LatentFeature: Update Multi-pcs->tum").draw(self.pc_list, None, blocking=True, origin_size=0.5)
        
    
    def knn_filter(self, samples:Tensor):
        
        N = samples.shape[0]
        indices = torch.arange(N, dtype=self.cfg.dtype.index, device=self.cfg.device)
        buffer_size = 1000000
        points =samples[:,:3]
        point_index = torch.ones(buffer_size, dtype=self.cfg.dtype.index, device=self.cfg.device) # [buffer_size]
         
        #Query
        query = samples[:,:3]
        query_grid = point2grid(query, self.resolution) # [N,3]
        query_hash = hash(query_grid, buffer_size) # [N]
        point_index[query_hash] = indices
        
        # Query-Neighbors
        neighbor_grid, _ = create_neighbors_grid(self.cfg.query.nvoxels_radius, self.cfg.query.ext_radius)
        query_neighbor_grid = query_grid[...,None,:] + neighbor_grid #[N,K,3]
        query_neighbor_hash = hash(query_neighbor_grid, buffer_size) # hash index [N,K] in (-infinity, +infinity)
        query_neighbor_idx = point_index[query_neighbor_hash] # [N,K] point index (valid index >= 0, invliad index = -1)
        
        queried_points = points[query_neighbor_idx] # [N,K,3]
        centroided_neighbors = queried_points[...,:3] - query.reshape(-1, 1, 3)
        neighbor_dist = centroided_neighbors.norm(p=2, dim=-1) # [N,K]
        
        valid_knn_mask = (query_neighbor_idx >= 0)
        valid_dist_mask = (neighbor_dist < self.resolution*(self.cfg.query.nvoxels_radius) + self.cfg.query.ext_radius)
        mask = valid_knn_mask & valid_dist_mask # [N,K]
        query_knn_count = mask.sum(dim=-1)
        valid_idx = (query_knn_count >= self.cfg.query.knn) # at least having k neighbors
        # print(f'    > knn_filter: N={N} -> filterd = {valid_idx.sum()}')
        return valid_idx
    '''
    Input: points [N,3]
    Output: points_neighbors [N,K,3]
    '''
    def query_neighbors(self,   query_xyz:Tensor, # [N, 3]
                                nvoxels_radius:int = 1, #[1,2,..]
                                ext_radius:float=0.0, # in [0..1)
                                is_temporal: bool = True,
                       ): # return [N*K,3]
        
        neighbor_grid, _ = create_neighbors_grid(nvoxels_radius, ext_radius)
        query_grid = point2grid(query_xyz, self.resolution) # [N,3]
        query_neighbor_grid = query_grid[...,None,:] + neighbor_grid #[N,K,3]
        query_neighbor_hash = hash(query_neighbor_grid, self.buffer_size) # hash index [N,K] in (-infinity, +infinity)
        
        query_neighbor_idx = self.point_index[query_neighbor_hash] # [N,K] point index (valid index >= 0, invliad index = -1)
         
        self.max_dist = math.sqrt(3)*(nvoxels_radius+1)*self.resolution
            
        if self.cfg.verbose >= 2: # Debug
            invalid = (query_neighbor_idx == -1).sum()
            valid = (query_neighbor_idx >= 0).sum()
            self.logger.log(2, f'  LatentFeature->query_neighbors: query_xyz={query_xyz.shape},  neighbor_idx={query_neighbor_idx.shape}, valid={valid}, invalid={invalid}, ratio={valid/(valid+invalid):0.2f}, total={valid+invalid}')
        
        if is_temporal:
            cur_acc_dist = self.travel_dist[self.frame_id].item()
            query_neighbor_ts = self.timestamp[query_neighbor_idx] # [N,K]
            diff_query_neighbor_travel_dist = cur_acc_dist  - self.travel_dist[query_neighbor_ts] # [1] - [N, K] = [N, K]
            diff_travel_dist_threshold = self.cfg.local_frame.radius*self.cfg.local_frame.travel_dist_ratio
            mask = (diff_query_neighbor_travel_dist >= diff_travel_dist_threshold) # neighbor_idx == -1 new point
            query_neighbor_idx[mask] = -1 # mask out invalid points
            self.logger.log(2, f'    > mask={mask.shape}, invalid_points={(mask == True).sum()}') 
           
        queried_points = self.points[query_neighbor_idx] # [N,K,3]
        centroided_neighbors = queried_points[...,:3] - query_xyz.reshape(-1, 1, 3) # [N,K,3] - [N,1,3] = [N,K,3]: make query points becomes centroids for their neighbors
        # dist = torch.norm(centroided_neighbors, p=2, dim=-1) # [N,K] -> cause in-place problems
        # print(centroided_neighbors)
        neighbor_sdist = torch.sum((centroided_neighbors)**2, dim=-1) # [N,K]
        neighbor_sdist[query_neighbor_idx == -1] = self.max_dist**2
        self.logger.log(3, f'    > number valid points in neighb_idx 1 = {(query_neighbor_idx !=-1).sum()} / {query_neighbor_idx.reshape(-1).shape[0]}, ratio = {(query_neighbor_idx !=-1).sum()/query_neighbor_idx.reshape(-1).shape[0]:0.2f}')
        query_neighbor_idx[neighbor_sdist >= self.max_dist**2] = -1 # mask out points far from its centroid
        self.logger.log(3, f'    > number valid points in neighb_idx 2 = {(query_neighbor_idx !=-1).sum()} / {query_neighbor_idx.reshape(-1).shape[0]}, ratio = {(query_neighbor_idx !=-1).sum()/query_neighbor_idx.reshape(-1).shape[0]:0.2f}')
        return neighbor_sdist, query_neighbor_idx # [N,K], [N,K]
    
    def query_features(self, query:Tensor,  # [N, F]
                       training=True,
                       from_id:int=-1, # with repsect to data from_id to to_id
                       to_id:int=-1,
                       ): 
        if self.print_tracking:
            self.logger.log(1, f'  LatentFeature->query_features: query_xyz={query.shape} in all points: self.points = {self.points.shape}')
        # Read configurations
        knn = self.cfg.query.knn
        feature_dim = self.cfg.feature.feature_dim
        N = query.shape[0]
        
        nvoxels_radius, ext_radius = self.cfg.query.nvoxels_radius, self.cfg.query.ext_radius
        neighbor_sdist, neighbor_idx = self.query_neighbors(query[:,:3], nvoxels_radius, ext_radius) # [N,K] indices
        neighbor_sdist[neighbor_idx == -1] = 9e3 # assign big value to sort to invalid points
        sorted_sdist, sorted_sdist_idx = torch.sort(neighbor_sdist, dim=-1) # [N,K]
        sorted_idx = neighbor_idx.gather(dim=1, index=sorted_sdist_idx) # select from sorted distance index
        knn_sdist = sorted_sdist[:,:knn] #[N,knn]
        knn_idx = sorted_idx[:,:knn] # [N,knn]  neighbors indices with k-nearest neighbors
        mask = (knn_idx >= 0) # vaild index mask [N,knn]
        
        if (from_id >= 0 and to_id >= 0) and (from_id <= to_id and to_id <= self.frame_id):
            ts_knn = self.timestamp[knn_idx]  # [N, knn]
            from_id, to_id = 1, 2
            ts_period_knn_mask = (from_id <= ts_knn) & (ts_knn <= to_id)
            # NN1 = mask.sum()
            mask = mask & ts_period_knn_mask # mask out neighbors invalid timestamps
            # NN2 = mask.sum()
            # print(f'    > NN1={NN1}, NN2={NN2}')
            # masked_points = self.points[knn_idx][mask].reshape(-1,3)
            # Vis('Period Points').draw([self.points[:,:3], masked_points[:,:3]],
            #                           [Vis.green, Vis.red], origin_size=3.0, blocking=True)
            # exit()

        if self.cfg.visual_debug and nvoxels_radius == 2:
            test_idx = 4 # selected point index in query
            p = query[test_idx].view(-1, self.input_dim)
            mask0 = (neighbor_idx >= 0) # [N,K]
            neighbor_count = mask0.sum(dim=-1)
            print_var(mask0, "mask0")
            print_var(neighbor_count[0:10], "neighbor_count", val=True)
            neighbors = self.points[neighbor_idx] # [N, K, 3]
            neighbors[~mask0] = p # suppress invalid points to p
            neighbors_test = neighbors[test_idx] # [K, 3]
            
            neighbors_knn = self.points[knn_idx] # [N, knn,3]
            neighbors_knn[~mask] = p # suppress invalid points to p
            neighbors_test_knn = neighbors_knn[test_idx] # [knn, 3]
            
            vis_lines = Vis.get_3d_grid(p,
                                        vs=self.resolution,
                                        r=self.cfg.query.nvoxels_radius,
                                        color=Vis.black)
            Vis(f"LatentFeature->query_features: FrameId={self.frame_id}").draw(list_tensors = [neighbors_test, neighbors_test_knn, p], 
                                                                          list_colors= [Vis.blue, Vis.green, Vis.red],
                                                                          blocking=True,
                                                                          origin_size=0.0,
                                                                          open3d_geometries=[vis_lines])

        self.logger.log(2, f'  LatentFeature->query_features: mask={mask.shape}, knn_idx_mask- = {knn_idx[mask].shape}, valid = {mask.sum().item()} out of {mask.shape[0]*mask.shape[1]}')
        
        # Get learned features
        query_features = torch.zeros(N, knn, feature_dim, device=self.cfg.device, dtype=self.cfg.dtype.point) # [N, knn, F]
        query_features[mask] = self.learned_features[knn_idx[mask]]
        
        # Neighbor geometry features
        neigbors = self.points[knn_idx] # [N,knn,input_dim]        
        neighbor_xyz_vec = query[:,:3].view(-1, 1, 3) -  neigbors[...,:3] # [N,1,input_dim]-[N,knn,input_dim]=[N,knn,input_dim]: centerlized at queried points
        neighbor_xyz_vec[~mask] = torch.zeros(1, 3).to(query) # assign [0,0,0..] for invalid neighbor vectors
        
        if query.shape[1] == 6: # XYZ-RGB 
            neighbor_color_vec = query[:,3:6].view(-1, 1, 3) -  neigbors[...,3:6]
            neighbor_color_vec[~mask] = torch.zeros(1, 3).to(query)
        else: # query with XYZ only, but point input_dim=6, copy from points
            if self.points.shape[1] == 6:
                neighbor_color_vec = neigbors[...,3:6]
                neighbor_color_vec[~mask] = torch.zeros(1, 3).to(query)
            else:
                neighbor_color_vec = torch.zeros(neigbors.shape[0], knn, 0).to(query)
        
        # Learned features and neighor features
        feature_vector = torch.cat([query_features, neighbor_xyz_vec, neighbor_color_vec], dim=-1) # [N,knn,F+input_dim]
        
        knn_count = (knn_idx >= 0).sum(dim=-1)
        weight_vector = 1.0/(knn_sdist + self.cfg.esp) # [N,knn] prevent divided by zeros
        # weight_vector[~mask] = eps            # [N,knn]
        # weight_vector[knn_count == 0] = eps   # [N,knn]
        weight_row_sums = torch.sum(weight_vector, dim=-1) #[N,1]
        w_mask = (weight_row_sums != 0)
        weight_vector[w_mask] = weight_vector[w_mask]/weight_row_sums[w_mask].unsqueeze(-1).tile(1,knn) #[N,knn]
        
        # weight_vector[~mask] = 0.0 
        with torch.no_grad(): # No gradient calculation 
            if training:
                knn_idx[~mask] = 0
                self.certainties.scatter_add_(dim=0, index=knn_idx.flatten(), src=weight_vector.flatten()) 
                if self.cfg.verbose >= 3:
                    self.logger.log(3, f'    > feature_vector = {feature_vector.shape}, weight_vector={weight_vector.shape}')
                    a,b = (self.certainties < 1.0).sum(), self.certainties.shape[0]
                    self.logger.log(3, f'    > valid_certainties = {a}/{b}, ratio={a/b: 0.2f}')
        
        weight_vector = weight_vector.unsqueeze(-1)  # [N, K, 1]
        # weight_vector = torch.ones_like(weight_vector)
        feature_vector = torch.sum(feature_vector * weight_vector, dim=1) #[N,K,F+input_dim]*[N,K,1] ->[N,K,F+input_dim] -sum-> [N,F+input_dim]
        
        if self.mode == 2:
            fvm=feature_vector.mean(dim=0)
            txt =''
            for i, f in enumerate(fvm):
                if i==10: # end
                    txt += f'{f:+0.4f}'
                else:
                    txt += f'{f:+0.4f}, '
            self.logger.write_track_feature(txt)
                
        if self.cfg.visual_debug: #  Visualize query and their neighbors with edges
        # if True:    
            valid_neigbors = neigbors.clone()
            valid_neigbors[~mask] = torch.zeros(1, self.cfg.feature.input_dim).to(neigbors) # mask invalid neighbors as zero
            n = 100
            # idx = torch.randperm(N)[0:n] # Choose n out of N randomly
            idx = torch.arange(0, n) # choose first n points of out N
            view_query = query[idx, ...] # [n,input_dim]
            view_query = view_query[:,:3] # [n,3]
            view_neighbors = valid_neigbors[idx,...] # [n,knn,input_dim]
            view_neighbors = view_neighbors[...,:3] # [n,knn,3]
            list_line_sets = Vis.get_neighbor_lines(view_query, view_neighbors, color=Vis.blue)
            Vis(f"LatentFeature->query_features: Edges between neighbors at frame_id={self.frame_id}").draw(list_tensors = [view_neighbors.view(-1,3), view_query], 
                                                                          list_colors= [Vis.green, Vis.red],
                                                                          blocking=True,
                                                                          origin_size=0.0,
                                                                          open3d_geometries=list_line_sets)
            
            exit()
        
        return feature_vector, weight_vector, knn_count
    
    
    def query_certainty(self, query_xyz: Tensor, # [N,3] 
                        ):
        self.logger.log(2, f'  LatentFeature->query_certainty: query_xyz={query_xyz.shape}')
        _, neighbor_idx = self.query_neighbors(query_xyz, nvoxels_radius=1, ext_radius=0.0) # [N,1] indices
        certainties = self.certainties[neighbor_idx]  # [N,1]
        certainties[neighbor_idx < 0] = 0.0 # [N,1]
        certainties = certainties.reshape(-1) # [N]
        return certainties