# @author    Tuan Dang   
# Email: tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

import numpy as np
from rich import print
from numpy import array, ndarray

class VirtualCamera():
    def __init__(self, image_width:int, image_height:int, 
                 fx:float, fy:float, 
                 cx:float, cy:float
            ):
        self.width, self.height = image_width, image_height
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        # self.intrinsic = np.eye(3)
        # self.intrinsic[0, 0] = fx
        # self.intrinsic[1, 1] = fy
        # self.intrinsic[:2, 2] = np.array([cx, cy])
        
    def set_camera_pose(self, pose: ndarray): # [4,4] # camera pose 
        self.extrinsic = pose # camera pose
        
    def raycasting(self, pose: ndarray, #[4,4] camera pose (extrinsic matrix)
                   points: ndarray=None #[N,F] where F=3 : poitns only, for F=6 with color
        ):
        
        print(f'raycasting: points={points.shape}')
        colors = None
        if points.shape[1] == 6:
            colors = points[:,3:6]
        
        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        homo_points = np.hstack((points[:,:3], np.ones((points.shape[0], 1)))) # [N,3] -> [N,4]
        # Normally: A[4,4]@P[4,N] -> [4,N]
        cam_points = (pose @ homo_points.T).T[:, :3]  # [N]
        
        # Points
        mask = cam_points[:,2] < 0
        cam_points = cam_points[mask] #  [N] Keep points with z > 0 (depen on how we present camera)
        
        u = (cam_points[:, 0] * fx / cam_points[:, 2]) + cx # N1
        v = (cam_points[:, 1] * fy / cam_points[:, 2]) + cy # N1
        depth = cam_points[:, 2] # [N1]
        if colors != None:
            colors = colors[mask]
        
        # Filter points that fall within the image bounds
        valid_pixels = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height) #[N2]
        u = u[valid_pixels].astype(int) # [N2]
        v = v[valid_pixels].astype(int) # [N2]
        depth = depth[valid_pixels]     # [N2]
        if colors != None:
            colors = colors[valid_pixels]
    
        depth_image = np.full((self.height, self.width), np.inf)
        rgb_image = np.full((self.height, self.width, 3), 1.0)
        # rgb_image = np.zeros((self.height, self.width, 3))
        print(f'raycasting: colors={colors.shape}: {colors[:10]}')
        for i in range(len(u)):
            depth_image[v[i], u[i]] = min(depth_image[v[i], u[i]], depth[i]) # choose  the point close to camera if the ray through multiple points
            if colors != None:
                rgb_image[v[i], u[i], :3] = colors[i]
        
        # depth_image =  np.fliplr(depth_image)
        return np.fliplr(depth_image), np.fliplr(rgb_image)

    @staticmethod
    def get_extrinsic_from_eye_lookat_up(eye: ndarray, #[3] camera position 
                                         lookat: ndarray, #[3] where camera look at 
                                         up: ndarray): #[3] camera up vector
        forward = (eye - lookat) # direction torward eye : vec(A->B)= B-A:  z-axis
        forward /= np.linalg.norm(forward) # normalize (we need direction only)
        
        right = np.cross(up, forward) #  [3] x-axis
        right /= np.linalg.norm(right)
        
        new_up = np.cross(forward, right) # [3]  y-axis
        new_up /= np.linalg.norm(new_up)  
        
        rotation_matrix = np.stack([right, new_up, forward], axis=1) # [3,3]
        translation_vector = -rotation_matrix.T@eye
        
        extrinsic = np.eye(4)
        extrinsic[:3,:3] = rotation_matrix
        extrinsic[:3,3] = translation_vector
        return extrinsic
       
if __name__ == "__main__":
    vc = VirtualCamera(640, 480, 525, 525, 320, 240)
    extrinsic = VirtualCamera.get_extrinsic_from_eye_lookat_up(eye=array([0.0, 0.0, 5.0]), 
                                                               lookat=array([0.0, 0.0, 0.0]), 
                                                               up=array([0.0, 1.0, 0.0]))
    print(extrinsic)