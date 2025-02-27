# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2024 Tuan Dang, all rights reserved

import open3d as o3d
from torch import tensor
import time
from torch import Tensor
import torch
import numpy as np
import matplotlib.colors as mcolors
import seaborn as sns
import os
from typing import List

class Visualizer:
    # Static properties     
    red   = np.array([1.0, 0.0, 0.0])
    green = np.array([0.0, 1.0, 0.0])
    blue  = np.array([0.0, 0.0, 1.0])
    black = np.array([0.0, 0.0, 0.0])
    colors = sns.color_palette("Set2", 8) # seaborn colors
    css4_colors = [mcolors.to_rgb(c) for c in mcolors.CSS4_COLORS] #CSS4 colors: https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors
    #names = [c for c in mcolors.CSS4_COLORS]
    
    def __init__(self, title='SIMN', logger=None):
        self.title = title
        self.window_created = False
        self.frame_id = 0
        self.logger = logger
       
    '''
    Draw continious/one-shoot frame within one window (non-blocking/blocking)
    '''
    def draw(self, list_pc, # List of [P, F] 
             list_colors=None, 
             blocking=True, # one time draw if true, window created every time  
             origin_size = 0.2, 
             open3d_geometries = None, # Open3d
             save_image=False, # Save image every frame in non-blocking mode if true
             save_pcd_rate = 30
        ):
        
        if self.window_created == False:
            self.window_created = True # windows created for continuously draw
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.title, width=1080, height=800, left=50, top=50, visible=True)
            
        self.vis.clear_geometries()
        N = len(list_pc)
        pcd = None
        for i in range(N):
            points = list_pc[i]
            if list_colors != None and len(list_colors) == N:
                color = list_colors[i]
            else:
                color = None
            pcd = self.tensor_to_pcd(points, color)
            self.vis.add_geometry(pcd)
            
        if origin_size > 0:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=origin_size, origin=[0, 0, 0])
            self.vis.add_geometry(coordinate_frame)
        
        if isinstance(open3d_geometries, List):    
            for g in open3d_geometries:
                self.vis.add_geometry(g)
        
        if blocking:
            self.vis.run()
        else: # None-blocking
            self.vis.poll_events()
            self.vis.update_renderer()
            if save_image:
                screenshoot_dir = f'{self.logger.dir}/screenshoot'
                pcd_dir = f'{self.logger.dir}/pcd'
                os.makedirs(screenshoot_dir, exist_ok=True)
                os.makedirs(pcd_dir, exist_ok=True)
                self.capture_screen(f'{screenshoot_dir}/screenshoot_{self.frame_id:04d}.png')
                if self.frame_id % save_pcd_rate == 0:
                    o3d.io.write_point_cloud(f'{pcd_dir}/pcd_{self.frame_id:04d}.pcd', pcd)
            self.frame_id += 1

    '''
    Point tensor to open3d data structure
    '''
    def tensor_to_pcd(self, points: Tensor=None, color: np.ndarray = None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3].detach().cpu().numpy())   
        if points.shape[1] == 6: #XYZ-RGB
            pcd.colors = o3d.utility.Vector3dVector(points[:,3:6].detach().cpu().numpy())   
        else:
            if  isinstance(color, np.ndarray): # color.shape = [1,3]
                pcd.paint_uniform_color(color)
        return pcd
    
    '''
    Save images
    '''
    def capture_screen(self, path):
        self.vis.capture_screen_image(path)
    
    def destroy(self):
        if self.window_created:
            self.vis.destroy_window()
        self.window_created = False
    
    @staticmethod
    def get_camera_frustum(camera, pose, depth=1.0, c=1):
        width, height = camera.width, camera.height
        fx, fy  = camera.fx, camera.fy
        cx, cy  = camera.cx, camera.cy
        # Define frustum corners in camera coordinate space
        z_far = 1.0 * depth  # Frustum depth
        corners = np.array([
            [0, 0, 0],  # Camera center
            [-(cx) / fx * z_far, -(cy) / fy * z_far, z_far],  # Top-left
            [(width - cx) / fx * z_far, -(cy) / fy * z_far, z_far],  # Top-right
            [(width - cx) / fx * z_far, (height - cy) / fy * z_far, z_far],  # Bottom-right
            [-(cx) / fx * z_far, (height - cy) / fy * z_far, z_far],  # Bottom-left
        ])
        
        # Transform corners to world space using pose
        ones = np.ones((corners.shape[0], 1))
        corners_homo = np.hstack((corners, ones))
        corners_world = (pose @ corners_homo.T).T[:, :3]

        # Define lines connecting corners
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # From camera center to frustum corners
            [1, 2], [2, 3], [3, 4], [4, 1],  # Frustum edges
        ]
        
        # Create line set
        if c == 1:
            colors = [ Visualizer.green, Visualizer.green, Visualizer.green, Visualizer.green, # front
                    Visualizer.red, Visualizer.red, Visualizer.red, Visualizer.red, #edges
                  ]
        elif c == 2:
            colors = [ Visualizer.blue, Visualizer.blue, Visualizer.blue, Visualizer.blue, # front
                    Visualizer.blue, Visualizer.blue, Visualizer.blue, Visualizer.blue, #edges
                  ]
        elif c == 3:
            colors = [Visualizer.red, Visualizer.red, Visualizer.red, Visualizer.red, #font 
                      Visualizer.red, Visualizer.red, Visualizer.red, Visualizer.red, # edges 
                  ]
        else:
            colors = [Visualizer.red, Visualizer.green, Visualizer.green, Visualizer.green, #font 
                      Visualizer.red, Visualizer.green, Visualizer.green, Visualizer.green, # edges 
                  ]
            
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_world)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    
   
    #==================================================================
    @staticmethod
    def get_neighbor_lines(query: Tensor, # [N,3] 
                  neighbors: Tensor, # [N, knn, 3]  
                  color=[0,0,0]) -> List:
        if query.shape[0] != neighbors.shape[0]:
            print(f'ERROR: get_neighbor_lines')
            exit()
        line_sets = []
        for i in range(query.shape[0]):
            source_point = query[i].view(-1, 3) # [1,3]
            dest_points = neighbors[i].view(-1, 3) # [knn, 3]
            mask = (dest_points != 0) # [knn, 3] valid
            sum_mask = torch.sum(mask, dim=-1) # [knn]: sum x,y,z each point
            mask_knn = (sum_mask != 0)  #[knn] sum(x,y,z) != 0 as valid points
            dest_points = dest_points[mask_knn]
            if dest_points.shape[0] == 0:
                continue
            points = torch.cat([source_point, dest_points], dim=0) # [1+knn, 3]
            points_np = points.detach().cpu().numpy()
            lines = np.zeros((points_np.shape[0]-1, 2), dtype=np.int16)
            for j in range(points_np.shape[0]-1):
                lines[j] = np.array([0, j+1])
            colors = [color for _ in range(len(lines))]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points_np)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            line_sets.append(line_set)
        return line_sets
    
    @staticmethod
    # draw the 3d-grid around point p with radius in number of cells
    def get_3d_grid(p: Tensor, #[1, 3]
                    vs=0.3, r=2, color=None):
        p = p.detach().cpu().numpy().reshape(-1)
        x, y, z = p
        points = []
        lines = []
        colors = []
        count = 0
        for zi in range(-r, r+1): # draw multiple planes xy along z-axis
            zv = z + zi*vs
            for xi in range(-r, r+1):
                xv = x + xi*vs # value of x
                ys = y - vs*r # y start
                ye = y + vs*r # y end
                points.append([xv, ys, zv])
                points.append([xv, ye, zv])
                lines.append([count*2, count*2+1])
                colors.append(color)
                count += 1
                
            for yi in range(-r, r+1):
                yv = y + yi*vs # value of y
                xs = x - vs*r # x start
                xe = x + vs*r # x end
                points.append([xs, yv, zv])
                points.append([xe, yv, zv])
                lines.append([count*2, count*2+1])
                colors.append(color)
                count += 1
                
        for xi in range(-r, r+1): # draw multiple planes yz along x-axis
            xv = x + xi*vs
            for yi in range(-r, r+1):
                yv = y + yi*vs # value of y
                zs = z - vs*r # z start
                ze = z + vs*r # z end
                points.append([xv, yv, zs])
                points.append([xv, yv, ze])
                lines.append([count*2, count*2+1])
                colors.append(color)
                count += 1
        
        points = np.array(points).reshape((-1,3))
        lines = np.array(lines).reshape((-1,2))
        colors = np.array(colors).reshape((-1,3))
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
            
    