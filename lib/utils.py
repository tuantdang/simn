# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2024 Tuan Dang, all rights reserved

from rich import print # print with color format
import torch
from torch import Tensor, tensor
import numpy as np
from typing import List
import math


def point2grid( points: Tensor, # [N,3]
                resolution: float = None
                ):
    point_xyz = points[:,0:3]
    return (point_xyz / resolution).floor().to(device=points.device, dtype=torch.int64) 
    
def hash(grid:Tensor, # [...,3]
         buffer_size,
        ):
    primes =  tensor([70001369, 80000201, 90002147], # X, Y, Z prime numbers
                         device=grid.device, dtype=torch.int64) # Size [3]
    # primes =  tensor([100000007, 100000037, 100000039], # X, Y, Z prime numbers
                        #  device=grid.device, dtype=torch.int64) # Size [3]
    h=torch.fmod((grid*primes).sum(dim=-1), #grid[N]=(x*prime1+y*prime2+z*prime3)%buffer_size
                    int(buffer_size)) # [...]
    return h

def create_neighbors_grid(nvoxels_radius:int = 1, #[1,2,..]
                              ext_radius:float=0.0, # in [0..1)
                              dev='cuda'
                            ):
        # Create 3D-Grids
        indices = torch.arange(-nvoxels_radius, nvoxels_radius + 1, # array [-nvoxels_radius:nvoxels_radius] step=1
                               device=dev, dtype=torch.int64) # size Ni
        grid = torch.meshgrid(indices, indices, indices, indexing='ij') #cubic(grid_x[Ni, Ni, Ni], grid_y[Ni, Ni, Ni], grid_z[Ni, Ni, Ni])
        grid = torch.stack(grid, dim=-1) # [Ni, Ni, Ni, 3]
        grid = grid.reshape(-1, 3); # [Ni x Ni x Ni, 3]
        dist = torch.norm(grid.to(dtype=torch.float32), p=2, dim=-1)  # [Ni x Ni x Ni]
        neighors_grid = grid[dist < (nvoxels_radius+ext_radius)] # [K, 3] Keep voxels inside the sphere 
        K = neighors_grid.shape[0]
        return neighors_grid, K
    
def voxel_down_sample(pc: Tensor, voxel_size: float, verbose=0):
        points = pc[:,0:3] # get XYZ only
        _quantization = 1000  # if change to 1, then it would take the first (smallest) index lie in the voxel
        offset = torch.floor(points.min(dim=0)[0] / voxel_size).long()
        grid = torch.floor(points / voxel_size)
        center = (grid + 0.5) * voxel_size
        dist = ((points - center) ** 2).sum(dim=1) ** 0.5
        dist = (dist / dist.max() * (_quantization - 1)).long()  # for speed up # [0-_quantization]

        grid = grid.long() - offset
        v_size = grid.max().ceil()
        grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size
        unique, inverse = torch.unique(grid_idx, return_inverse=True)
        idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        offset = 10 ** len(str(idx_d.max().item()))
        idx_d = idx_d + dist.long() * offset
        idx = torch.empty(unique.shape, dtype=inverse.dtype, device=inverse.device).scatter_reduce_(
            dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
        idx = idx % offset
        ret = pc[idx] # [N,F] (F>=3)
        if verbose >= 2: 
            print(f'  Downsampling: pc={pc.shape} -> ret={ret.shape} with voxel_size={voxel_size}')
        return ret, idx

def crop_frame(points:Tensor, min_z=-3.0, max_z=100.0, min_range=2.75, max_range=100.0):
    dist = torch.norm(points[:, :3], dim=1)
    filtered_idx = ((dist > min_range) & (dist < max_range) & (points[:, 2] > min_z) & (points[:, 2] < max_z))
    points = points[filtered_idx]
    return points

def make_tabs(n):
    tabs = ""
    for _ in range(n):
        tabs = tabs + "    "
    return tabs

def print_item(obj, depth:int, lst:List):
    for k, v in obj.__dict__.items():
        if hasattr(v, '__dict__'):
            txt = f'{make_tabs(depth)}{k}'
            lst.append(txt)
            print_item(v, depth+1)
        else:
            txt = f"{make_tabs(depth)}{k}: {v}"
            lst.append(txt)

def object2text(obj): 
    lst_txt = []
    print_item(obj, 0, lst_txt)
    all_txt = ""
    for i, txt in enumerate(lst_txt):
        if i != len(lst_txt) - 1: # not end list
            all_txt += txt + "\n"
        else: #end of list
            all_txt += txt
    return all_txt


# Print atributes and values in an object recursively
def print_object(obj): 
    print("==========Start Object====================")
    print(object2text(obj))
    print("==========End Object====================")

def print_var(var, var_text, val=False):
    if val:
        print(f'    > {var_text}={var}')
        return
    
    if var is None:
        print(f'    > {var_text}={None}')
    else:
        if isinstance(var, torch.Tensor) or isinstance(var, np.ndarray):
            print(f'    > {var_text}={var.shape}')
        else:
            print(f'    > {var_text}={var:0.2f}')

def detect_nan(values: Tensor, message=''):
    nan_mask = torch.isnan(values)
    has_nan = torch.any(nan_mask)
    if has_nan: # registration > query
        print(f'Detect Nan: {message} : {values.shape}')
        print(values[0])
        import traceback
        traceback.print_stack()
        exit()