# @author    Tuan Dang   
# Email: tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2024 Tuan Dang, all rights reserved


import torch
import numpy as np
import open3d as o3d
from torch import Tensor, tensor
from rich import print
import time
from torchmcubes import marching_cubes, grid_interp # Refer at https://github.com/tatsy/torchmcubes

from lib.utils import print_var, hash, point2grid, create_neighbors_grid
import warnings

warnings.filterwarnings("ignore")
import math

from lib.visualizer import Visualizer as Vis

class MarchingCube():
    def __init__(self, cfg=None):
        self.cfg = cfg
       


    def create_mesh_from_pc(self, pc:Tensor, # [N,3]
                              sdf = None,# [N]
                              voxel_size=0.1, threshold=0.1):
        xyz_points = pc[:,:3]
        colors = None
        if pc.shape[1] == 6:
            alpha = torch.zeros((xyz_points.shape[0], 1)).to(xyz_points) # [N,1]
            colors = torch.cat([pc[:, 3:6], alpha], dim=-1).contiguous() # [N,3] concat [N,1] -> [N,4]
            
        print_var(xyz_points, 'xyz_points')
        min_bounds = torch.min(xyz_points, dim=0)[0] - voxel_size*2 # both ends (2)
        max_bounds = torch.max(xyz_points, dim=0)[0] + voxel_size*2 # both ends (2)
        
        print_var(min_bounds, 'min_bounds', val=True)
        print_var(max_bounds, 'max_bounds', val=True)

        grid_x, grid_y, grid_z = torch.meshgrid(
                torch.arange(min_bounds[0], max_bounds[0], voxel_size), # Nx
                torch.arange(min_bounds[1], max_bounds[1], voxel_size), # Ny
                torch.arange(min_bounds[2], max_bounds[2], voxel_size), # Nz
            ) # grid_x, grid_y, grid_z hold [Nx, Ny, Nz] indices
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1).to(pc) # [Nx*Ny*Nz, 3]
        
        print(f'grid_x = {grid_x.shape}, grid_points={grid_points.shape}')
        if sdf == None: # calculate SDF volums or calculate sdf value for each voxel as min distance to its neighbors
            t1=time.time()
            distances = torch.cdist(grid_points, xyz_points)    # [Nx*Ny*Nz, N]
            min_distances, _ = distances.min(dim=1)             # [Nx*Ny*Nz,1]
            volume = min_distances.reshape(grid_x.shape)        # [Nx,Ny,Nz]
            t2=time.time()
            print(f'create volume = {volume.shape}, time={t2-t1:0.3f} s')
        else: # SDF provided (ray casting, sensor, sampling)
            N = xyz_points.shape[0]
            Nx = torch.ceil((max_bounds[0] - min_bounds[0])/voxel_size).int().item()
            Ny = torch.ceil((max_bounds[1] - min_bounds[1])/voxel_size).int().item()
            Nz = torch.ceil((max_bounds[2] - min_bounds[2])/voxel_size).int().item()
            print('Grid size = ', Nx, Ny, Nz, f'vs grid_x = {grid_x.shape}')
            
            buffer_size = int(2.0e9)
            ref_point_index = torch.full((buffer_size,), -1, 
                                          dtype=torch.int64, device=xyz_points.device) # Filled with -1
            
            # update points:
            ref_grid = point2grid(xyz_points, voxel_size)
            ref_hash = hash(ref_grid,buffer_size )
            print_var(ref_grid, 'ref_hash')
            print_var(ref_hash, 'ref_hash')
            
            ref_point_index[ref_hash] = torch.arange(N,  
                                               dtype=torch.int64, device=xyz_points.device)
            

            # Vis("Grid-points").draw([grid_points], [Vis.red])
           
            neighbor_grid, K = create_neighbors_grid(2, 0.2, dev=xyz_points.device)
            print_var(neighbor_grid, 'neighbor_grid')
            print_var(grid_points, 'grid_points')
            query_grid = point2grid(grid_points, voxel_size) # [N,3]
            print_var(query_grid, 'query_grid')
            query_neighbor_grid = query_grid[...,None,:] + neighbor_grid #[N,K,3]
            print_var(query_neighbor_grid, 'query_neighbor_grid')
            query_neighbor_hash = hash(query_neighbor_grid, buffer_size) # hash index [N,K] in (-infinity, +infinity)
            print_var(query_neighbor_hash, 'query_neighbor_hash')
            query_neighbor_idx = ref_point_index[query_neighbor_hash] # [N,K] point index (valid index >= 0, invliad index = -1)
            print_var(query_neighbor_idx, 'query_neighbor_idx')
            idx = query_neighbor_idx
            
            print()
            query = grid_points
            print_var(query, 'query(grid_points)')
            queried_points = xyz_points[idx] # [N,K,3]
            # Vis("Quries-point").draw([query_points.view(-1, 3)], [Vis.red])
            # exit()
            
            print_var(queried_points, 'query_points')
            centroids = queried_points - query.view(-1, 1, 3)
            print_var(centroids, 'centroids')
            print()
            
            max_dist = math.sqrt(3)*(2+1)*voxel_size
            neighbor_sdist = torch.sum(centroids**2, dim=-1) # [N,K]
            neighbor_sdist[idx == -1] = max_dist**2
            idx[neighbor_sdist >= max_dist**2] = -1
            neighbor_sdist[idx == -1] = 9e3
            sorted_sdist, sorted_sdist_idx = torch.sort(neighbor_sdist, dim=-1) # [Nx*Ny*Nz,K]
            sorted_idx = idx.gather(dim=1, index=sorted_sdist_idx) # select from sorted distance index
            knn = 1
            knn_sdist = sorted_sdist[:,:knn] #[Nx*Ny*Nz,1] # select 1 close
            knn_idx = sorted_idx[:,:knn]
            print_var(knn_idx, 'knn_idx')
            sdf_knn = sdf[knn_idx] # [Nx*Ny*Nz, 1]
            print_var(sdf_knn, 'sdf_knn')
            # exit()
            
            mask = (knn_idx >= 0)
            print_var(mask.sum(), 'mask_sum', val=True)
            new_queried_points = xyz_points[knn_idx]
            print_var(new_queried_points, 'new_queried_points')
            # Vis("New Quried Points").draw([new_queried_points.view(-1, 3)], [Vis.red])
           
            mask_knn_count = mask.sum(dim=-1) # for each query point, count its valid neighbors
            mask_gp = mask_knn_count > 0
            print()
            print_var(mask_knn_count, 'mask_knn_count')
            print_var(mask_gp, 'mask_gp')
            print_var(mask_gp.sum(), 'mask_gp_sum', val=True)
            
            sdf_knn[~mask] = threshold*1.5
            volume = sdf_knn.reshape(grid_x.shape)
            
            if colors != None:
                colors = colors[knn_idx].view(-1,4)
                # colors = grid_interp(colors, grid_points)
            
        t1=time.time()
        verts, faces = marching_cubes(volume, threshold)
        verts = verts * voxel_size + min_bounds # [Nvertices, 3]
        verts_tensor = verts
        verts, faces = verts.detach().cpu().numpy(), faces.detach().cpu().numpy()
        # print(min_bounds.cpu().numpy())
        # exit()
        # verts = verts * voxel_size + min_bounds.cpu().numpy() # convert to oginal coordinates
        # verts = verts * voxel_size + np.array([0])
        
        t2=time.time()
        # print()
        # print(sdf_knn[~mask])
        # print()
        print_var(volume, "volume")
        print(f"marching cube: verts={verts.shape}, faces={faces.shape}, time: {t2 - t1:.3f} s")
        
        if colors != None:
            colors = colors.cpu().numpy()

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        # exit()
        if sdf != None:
            Vis("Reconstruction: with SDF and new queried points").draw([new_queried_points.view(-1, 3)], [Vis.red], 
                                        blocking=True,
                                        origin_size=0.5,
                                        open3d_geometries=[mesh])
        else:
            from liealgebra import LieAlgebra as la
            # T1 = la.axis_angle_to_transform([0.0, 1.0, 0.0], -90.0, dev=pc.device).cpu().numpy() # Oy
            # T2 = la.axis_angle_to_transform([0.0, 0.0, 1.0], 180.0, dev=pc.device).cpu().numpy() # Oz
            # mesh = mesh.transform(T2@T1)
            # mesh = mesh.translate([min_bounds[0].item(), min_bounds[1].item(), min_bounds[2].item()])
            Vis("Reconstruction: NO SDF").draw([pc[:,:3], verts_tensor], [Vis.red, Vis.green], 
                                        blocking=True,
                                        origin_size=3.0,
                                        open3d_geometries=[mesh])
        
        return verts, faces, colors
    

def test_with_dataset():
    from lib.config import Config
    from dataset.tumdataset import TumDataset
    from dataset.kittidataset import KittiDataset
    from logger import Logger
    from sampling import Sampling
    # cfg = Config(path="config/tum.yaml").config()
    cfg = Config(path="config/kitti.yaml").config()
    logger = Logger(cfg)
    # dataset = TumDataset(cfg, logger)
    dataset = KittiDataset(cfg, logger)
    sampler = Sampling(cfg, logger)
    mc = MarchingCube(cfg)
    while dataset.available():
        framid, pc = dataset.next_frame()
        # points, sdf, _ = sampler.dist_sampling(pc)
        # points = points[::1,:]
        # point
        points = pc[::2,:]
        # Vis('TUM').draw([points], [Vis.red])
        print(f'points = {points.shape}')
        vs = 1.5
        verts, faces, colors = mc.create_mesh_from_pc(points, None, vs, vs)
        
        # Visualize
        '''
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        # mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
        '''
        break
    
    
def test_marching_cube():
    # Grid data
    N = 128
    xs = np.linspace(-1.0, 1.0, N, endpoint=True, dtype="float32")
    ys = np.linspace(-1.0, 1.0, N, endpoint=True, dtype="float32")
    zs = np.linspace(-1.0, 1.0, N, endpoint=True, dtype="float32")
    zs, ys, xs = np.meshgrid(zs, ys, xs)

    # Implicit function (metaball)
    f0 = (xs - 0.35)**2 + (ys - 0.35)**2 + (zs - 0.35)**2
    f1 = (xs + 0.35)**2 + (ys + 0.35)**2 + (zs + 0.35)**2
    u = 4.0 / (f0 + 1.0e-6) + 4.0 / (f1 + 1.0e-6)

    rgb = np.stack((xs, ys, zs), axis=-1) * 0.5 + 0.5
    rgb = np.transpose(rgb, axes=(3, 2, 1, 0))
    rgb = np.ascontiguousarray(rgb)

    # Test
    u = torch.from_numpy(u)
    rgb = torch.from_numpy(rgb)
    u = u.cuda()
    rgb = rgb.cuda()

    t_start = time.time()
    verts, faces = marching_cubes(u, 15.0)
    colors = grid_interp(rgb, verts)
    t_end = time.time()
    

    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    colors = colors.detach().cpu().numpy()
    verts = (verts / (N - 1)) * 2.0 - 1.0  # Get back to the original space
    # visualize(verts, faces, colors)
    print(f"verts: {verts.shape}, faces: {faces.shape}, colors={colors.shape}, time: {t_end - t_start:.3f} s")
    
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    # mesh.vertex_colors = o3d.utility.Vector3iVector(colors)
    colors = np.random.rand(len(verts), 3)  # Random RGB colors for each vertex
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([mesh])

def sample_plane(normal, center, w=0.1, l=0.1, h=0.1, N=100):
  #ax + by + cz + d = 0
  normal /= torch.norm(normal)
  a, b, c = normal
  x0, y0, z0 = center
  d = -torch.dot(normal, center)
  x = torch.FloatTensor(N).uniform_(-w/2, w/2) + x0
  y = torch.FloatTensor(N).uniform_(-l/2, l/2) + y0
  z = torch.FloatTensor(N).uniform_(-h/2, h/2) + z0
  if a != 0:
    x = (d - b*y - c*z)/a
  elif b != 0:
    y = (d - a*x - c*z)/b
  elif c != 0:
    z = (d - a*x - b*y)/c
  pc = torch.cat([x[:,None], y[:,None], z[:,None]], dim=-1)
  return pc

def sample_box( Wx, Ly, Hz, N):
  A = tensor([0, 0, 0]) #origin
  B = tensor([Wx, 0, 0])
  C = tensor([Wx, Ly, 0])
  D = tensor([0, Ly, 0])
  E = tensor([0, 0, Hz])
  F = tensor([Wx, 0, Hz])
  G = tensor([0, Ly, Hz])
  H = tensor([Wx, Ly, Hz])
  keypoints =  torch.vstack([A.reshape(1,3), B.reshape(1,3), C.reshape(1,3), D.reshape(1,3), E.reshape(1,3), F.reshape(1,3), G.reshape(1,3), H.reshape(1,3)])
  npoints = int(N/6) + 1 # 6 planes

  pc1 = sample_plane(tensor([1, 0, 0]).float(), tensor([0, Ly/2, Hz/2]),  Wx, Ly, Hz, npoints)
  pc2 = sample_plane(tensor([1, 0, 0]).float(), tensor([-Wx, Ly/2, Hz/2]), Wx, Ly, Hz, npoints)
  pc3 = sample_plane(tensor([0, 1, 0]).float(), tensor([Wx/2, 0, Hz/2]), Wx, Ly, Hz, npoints)
  pc4 = sample_plane(tensor([0, 1, 0]).float(), tensor([Wx/2, -Ly, Hz/2]), Wx, Ly, Hz, npoints)
  pc5 = sample_plane(tensor([0, 0, 1]).float(), tensor([Wx/2, Ly/2, 0]), Wx, Ly, Hz, npoints)
  pc6 = sample_plane(tensor([0, 0, 1]).float(), tensor([Wx/2, Ly/2, -Hz]), Wx, Ly, Hz, N-npoints*5)

  pc =  torch.cat([pc1, pc2, pc3, pc4, pc5, pc6], dim=0)
  edge_index = tensor([[0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7],
                             [1,3,4, 0,2,5, 1,3,7, 0,2,6, 0,5,6, 1,4,7, 3,4,7, 2,5,6]]).long()
  N = pc.shape[0]
  sdf = torch.zeros((N,)).to(pc)
  return pc, keypoints, edge_index, sdf

def test_sampling():
    pc, keypoints, edge_index, sdf = sample_box(2, 2, 2, 1000*6)
    # Vis("Sampled points").draw([pc], [Vis.red])
    print(pc.device)
    mc = MarchingCube()
    # mc.create_mesh_from_pc(pc, sdf, 0.05, 0.05)
    mc.create_mesh_from_pc(pc.cuda(), None, 0.1, 0.1)
    

if __name__ == "__main__":
    test_with_dataset()
    # test_marching_cube()
    # test_sampling()