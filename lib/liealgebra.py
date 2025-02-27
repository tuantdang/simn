# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved


from torch import Tensor, tensor
import torch
from rich import print
import math
import time

class LieAlgebra:
    def __init__(self):
        pass
    
    @staticmethod
    # Typical points should represent P=[3,N] to apply transform T: P'=T@P  : [3,N]
    # But in our frame work points is represent as [N,3] therefore: P'=(T@P)^T = P^T @ T^T= Pt@T^T where Pt is [N,3]
    def transform(points: Tensor, # [N,3] 
                  T:Tensor, # [4,4]
                  homogeneous = True # True: faster
                  ):
        N= points.shape[0]
        if not homogeneous:
            R = T[:3,:3] # Rotation # [3,3]
            t = T[:3, 3].view(1,3) # Translation [1,3]
            new_points = points@R.T + t.tile(N,1) # [3,3]@[3.N] -> ([N,3]@[3,3])^T = [N,3]^T@[3,3]^T -> [N,3]
        else:
            homo_points = torch.cat([points, torch.ones((N,1), device=points.device)], dim=-1)
            new_points = homo_points @ T.T # [N,4]
            new_points = new_points[:,:3]  # [N,3]
        return new_points
    
    @staticmethod
    def skew(w) -> Tensor: # make skew matrix from axis-angle
        omega = torch.zeros(3, 3, device=w.device, dtype=w.dtype)
        omega[0, 1] = -w[2]
        omega[0, 2] = w[1]
        omega[1, 2] = -w[0]
        return omega - omega.T
    
    @staticmethod
    def build_axis_angle(axis:Tensor, angle:Tensor) -> Tensor: # complile axis and angle into angle_axis
        unit_vec = axis/axis.norm(p=2)
        return unit_vec*angle
    
    @staticmethod
    def extract_axis_angle(axis_angle:Tensor) -> Tensor: # seperate axis, angle from axis_angle
        angle = axis_angle.norm(p=2)
        if angle != 0:
            unit_axis = axis_angle/angle # unit_vec = w/angle
        else:
            unit_axis = axis_angle
        return unit_axis, angle # [3], rad
    
    @staticmethod
    def extract_axis_angle_batch(axis_angle:Tensor #[bs, 3]
                                 ) -> Tensor: # seperate axis, angle from axis_angle
        angle = axis_angle.norm(p=2, dim=-1) # [bs]
        mask = (angle != 0)
        unit_axis = axis_angle.clone()
        unit_axis[mask] = axis_angle[mask]/angle[mask].unsqueeze(-1).tile(1,3) # [bs,3] unit_vec = w/angle
        return unit_axis, angle # [bs, 3], [bs]
    
    @staticmethod
    def so3_exp_map(axis_angle: torch.Tensor)  -> Tensor : # Axis-angle to Rotation Matrix
        assert  torch.is_tensor(axis_angle) and axis_angle.shape == (3,), "Invalid so(3) format (should be in R^3)"
        angle = axis_angle.norm(p=2)
        if angle != 0:
            axis = axis_angle / angle
        else:
            axis = axis_angle / (angle + 1.0e-15)
        I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
        omega = LieAlgebra.skew(axis)
        R = I + torch.sin(angle)*omega +  (1.0 - torch.cos(angle))*(omega @ omega)
        return R
    
    @staticmethod
    def so3_log_map(R)  -> Tensor: #  Axis Angle (so3=R^3) to Rotation Matrix (SO3=R^4x4)
        assert torch.is_tensor(R) and R.shape == (3, 3), "Invalid rotation matrix, should be in R^(3x3)"
        
        angle = torch.acos((torch.trace(R) - 1) / 2)
        # print(f'so3_log_map: trace = {torch.trace(R)}, , angle={angle}')
        axis = tensor([R[2,1] - R[1,2], 
                       R[0,2] - R[2,0], 
                       R[1,0] - R[0,1]], device=R.device)
        sin_alpha = torch.sin(angle)
        
        if sin_alpha != 0:
            axis = axis/(2*sin_alpha) # why not w=angle*unit_vector: because axis=[0,0,0] -> angle mean nothing
        
        length = axis.norm(p=2, dim=-1) 
        if length != 0:
            unit_vec = axis/length
        else:
            unit_vec = tensor([0.0, 0.0, 0.0]).to(R)
        
        axis_angle = unit_vec*angle #[3]*[1]
        # print(f'so3_log_map: axis_angle={axis_angle}')
        return axis_angle  #  [3]
    
    @staticmethod
    def so3_log_map_batch(R: Tensor #[bs, 3,3]
                          ) -> Tensor: #  [bs, 3]
        # angle = torch.acos((R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) - 1) / 2) # [bs]
        # angle = torch.acos((torch.einsum('...ii', R) - 1) / 2) # [bs]
        angle = torch.acos((torch.vmap(torch.trace)(R) - 1) / 2) # [bs]
        # print(f'so3_log_map_batch: trace = {torch.vmap(torch.trace)(R)}, angle={angle}')
        
        axis = torch.cat([  (R[:,2,1] - R[:,1,2]).unsqueeze(-1), #[bs,1]
                            (R[:,0,2] - R[:,2,0]).unsqueeze(-1),  #[bs,1]
                            (R[:,1,0] - R[:,0,1]).unsqueeze(-1)], dim=1) # [bs,3]
        
        # print(f'angle = {angle.shape}, axis={axis.shape}')
        sin_alpha = torch.sin(angle)
        
        mask = (sin_alpha != 0)
        axis[mask] = axis[mask]/(2*sin_alpha[mask]).unsqueeze(-1).tile(1,3) # [bs,3]
        
        length = axis.norm(p=2, dim=-1) # [bs]
        
        mask = length > 0
        unit_vec = axis.clone()
        unit_vec[mask] = axis[mask]/length[mask].unsqueeze(-1).tile(1,3) # [bs,3]
        axis_angle = unit_vec*angle.unsqueeze(-1).tile(1,3)
        # print(f'so3_log_map_batch: axis_angle={axis_angle}')
        return axis_angle  #  [bs, 3]
    
    @staticmethod
    def se3_exp_map(p: Tensor) -> Tensor:  # se(3)=R^6 -> SE(3)=R^(4x4)
        assert  torch.is_tensor(p) and p.shape == (6,), "Invalid se(3) format (should be in R^6)"
        T = torch.eye(4, device=p.device, dtype=p.dtype)
        T[:3,:3] = LieAlgebra.so3_exp_map(p[:3]) # axis_angle to rotation matrix
        T[:3,3] = p[3:] # translation
        return T
    
    @staticmethod
    def se3_log_map(T: Tensor) -> Tensor: # SE(3)=R^(4x4)-> se(3)=R^6
        p = torch.zeros(6).to(T)
        p[:3] = LieAlgebra.so3_log_map(T[:3,:3])
        p[3:6] = T[:3,3]
        return p
    
    @staticmethod
    def rad2deg(rad):
        return (rad/math.pi)*180.0
    
    @staticmethod
    def deg2rad(deg):
        return (deg/180.0)*math.pi
    
    @staticmethod
    def view_transform(T:Tensor): # For logging and debugging
        axis_angle = LieAlgebra.so3_log_map(T[:3,:3]) # rotation matrix to axis_angle
        _, angle_rad = LieAlgebra.extract_axis_angle(axis_angle)
        angle_deg = LieAlgebra.rad2deg(angle_rad)
        tr_dist= T[:3,3].norm(p=2)
        tr_vec = T[:3,3].cpu().numpy()
        q = LieAlgebra.rot_to_quat(T[:3,:3]).cpu().numpy()
        txt = f'{angle_deg:+2.5f}, {tr_dist:+2.5f}, {tr_vec[0]:+2.3f}, {tr_vec[1]:+2.3f}, {tr_vec[2]:+2.3f}, {q[0]:+2.3f}, {q[1]:+2.3f}, {q[2]:+2.3f}, {q[3]:+2.3f}'
        return txt
    
    @staticmethod
    def view_transform2(T:Tensor): # For logging and debugging
        tr_vec = T[:3,3].cpu().numpy()
        q = LieAlgebra.rot_to_quat(T[:3,:3]).cpu().numpy()
        txt = f'{tr_vec[0]:2.3f} {tr_vec[1]:2.3f} {tr_vec[2]:2.3f} {q[0]:2.3f} {q[1]:2.3f} {q[2]:2.3f} {q[3]:2.3f}'
        return txt
    
    @staticmethod
    def axis_angle_to_transform(axis, # List of 3 or Tensor/Numpy  
                                angle:float, # radian
                                dev = 'cuda' 
                                ):
        T = torch.eye(4).to(dev) # Identity matrix
        axis_t = tensor(axis).to(dev) 
        angle_t = tensor(angle, device=dev) #
        axis_angle = LieAlgebra.build_axis_angle(axis_t, angle_t)
        rot = LieAlgebra.so3_exp_map(axis_angle) #
        T[:3,:3] = rot
        
        return T
    
    
    
    @staticmethod
    def rot_to_quat(r:Tensor # [3,3]
                ):
        trace = torch.trace(r)
        if trace > 0:
            w = 0.5*torch.sqrt(1+trace)
            x = (r[2,1] - r[1,2])/(4*w)
            y = (r[0,2] - r[2,0])/(4*w)
            z = (r[1,0] - r[0,1])/(4*w)
        else:
            id = torch.argmax(torch.diag(r))
            
            if id == 0:
                x = 0.5*torch.sqrt(1 + r[0,0] - r[1,1] - r[2,2])
                w = (r[2,1] - r[1,2])/(4*x)
                y = (r[0,1] + r[1,0])/(4*x)
                z = (r[0,2] + r[2,0])/(4*x)
            elif id == 1:
                y = 0.5*torch.sqrt(1 - r[0,0] + r[1,1] - r[2,2])
                w = (r[0,2] - r[2,0])/(4*y)
                x = (r[0,1] + r[1,0])/(4*y)
                z = (r[1,2] + r[2,1])/(4*y)
            else:
                z = 0.5*torch.sqrt(1 - r[0,0] - r[1,1] + r[2,2])
                w = (r[1,0] - r[0,1])/(4*z)
                x = (r[0,2] + r[2,0])/(4*z)
                y = (r[1,2] + r[2,1])/(4*z)

        return tensor([x, y, z, w]).to(r)

    @staticmethod
    def quat_to_rot(q: Tensor # [4]
                    ):
        x, y, z, w = q
        return  tensor([[1 - 2*(y**2 + z**2), 2*(x*y - w*z),         2*(x*z + w*y)],
                        [2*(x*y + w*z),       1 - 2*(x**2 + z**2),   2*(y*z - w*x) ],
                        [2*(x*z - w*y),       2*(y*z + w*x),         1 - 2*(x**2 + y**2)]
                    ]).to(q)
        
    