from torch import tensor, Tensor, min, max, abs
import torch
import time
from rich import print
from lib.logger import Logger
from lib.utils import print_var
import math
from lib.visualizer import Visualizer as Vis

class Sampling:
    def __init__(self, cfg=None, logger:Logger=None):
        self.cfg = cfg
        self.logger = logger

    ''''
    Form a ray PO as normal vector for each point P, where O is origin at local frame
    Sample  2D-points around P projected on either XY,YZ,XZ planes
    Calculate the remain axis
    '''
    
    def plane_sample(self, pc: Tensor, #[N,3]
            K=6, #sample K points for each point in points
            surface_range = 0.1, # square surface in meter
            ):
        points = pc[:,:3]
        dev = pc.device
        origin = tensor([0, 0, 0], dtype=torch.float32, device=dev) # local frame
        
        # Find the remain axis 
        pc_min_bound, _ = min(points[:, :3], dim=0) # get values [min_x, min_y, min_z]
        max_min_value, max_min_index = max(abs(pc_min_bound), dim=-1)
        mask = torch.abs(points[:,max_min_index]) > 0.05 # x cm from
        N0 =  points.shape[0]
        points = points[mask]
        N = points.shape[0]
        self.logger.log(1, f'   Sampling->sample: {N0} - > {N}')
        vec_po = origin.reshape(1,3).tile((N,1)) - points # [N,3]
        dist = torch.linalg.norm(vec_po, ord=2, dim=-1).reshape((N,1)).tile((1,3)) # [N,3]
        normals = vec_po/dist
        d = -torch.sum(normals*points, dim=-1, keepdim=True).tile((1, K)) #[N,K]
        
        if max_min_index == 2: # z-axis
            delta_xy = torch.rand((N, K, 2), device=dev)*surface_range*(-2) + surface_range # [-surface_range, surface_range], [N,K,2]
            # delta_xy = torch.normal(mean=0, std=0.1, size=(N, K, 2), device=dev)*surface_range*(-2) + surface_range
            xy = delta_xy + points[:,0:2].reshape(N, 1, 2).tile((1, K, 1))# xy around p1; [N,K,2]
            normal_xy =    normals[:,0:2].reshape(N, 1, 2).tile((1, K,1 )) #[N,K,2]
            z = -(torch.sum(normal_xy*xy, dim=-1) + d) # [N,K]  z = -(ax+by+d)/c, if c>0
            c = normals[:,2].reshape((N,1)).tile((1,K)) # [N,1]
            z = z/c # [N,K]
            sampled_points = torch.cat([xy, z.reshape(N, K,1)], dim=-1) # [N, K, 3]
        elif max_min_index == 0+0: # x-axis
            delta_yz = torch.rand((N, K, 2), device=dev)*surface_range*(-2) + surface_range # [-surface_range, surface_range], [N,K,2]
            # delta_xy = torch.normal(mean=0, std=0.1, size=(N, K, 2), device=dev)*surface_range*(-2) + surface_range
            yz = delta_yz + points[:,1:3].reshape(N, 1, 2).tile((1, K, 1))# xy around p1; [N,K,2]
            normal_yz =    normals[:,1:3].reshape(N, 1, 2).tile((1, K, 1)) #[N,K,2]
            x = -(torch.sum(normal_yz*yz, dim=-1) + d) # [N,K]  x = -(by+cz+d)/a, if a>0
            a = normals[:,0].reshape((N,1)).tile((1,K)) # [N,K]
            x = x/a # [N,K]
            sampled_points = torch.cat([x.reshape(N, K,1), yz ], dim=-1) # [N, K, 3]
        else: # y-axis
            self.logger.log(1, f'     > Y-axis is choosen, points[:,0:3:1]={points[:,0:3:2].shape}')
            delta_xz = torch.rand((N, K, 2), device=dev)*surface_range*(-2) + surface_range # [-surface_range, surface_range], [N,K,2]
            # delta_xy = torch.normal(mean=0, std=0.1, size=(N, K, 2), device=dev)*surface_range*(-2) + surface_range
            xz = delta_xz + points[:,0:3:2].reshape(N, 1, 2).tile((1, K, 1))# xz around p1; [N,K,2]
            normal_xz =    normals[:,0:3:2].reshape(N, 1, 2).tile((1, K, 1)) #[N,K,2]
            y = -(torch.sum(normal_xz*xz, dim=-1) + d) # [N,K]  y = -(ax+cz+d)/b, if b>0
            
            b = normals[:,1].reshape((N,1)).tile((1,K)) # [N,K]
            y = y/b # [N,K]
            x, z = xz[...,0], xz[...,1]
            sampled_points = torch.cat([x.reshape((N,K,1)), y.reshape((N,K,1)), z.reshape((N,K,1), ) ], dim=-1) # [N, K, 3] order [y,x,z]
            self.logger.log(1, f'     > Y-axis is choosen, xz={xz.shape}, y={y.shape}, sampled_points={sampled_points[0]}')

        return sampled_points.reshape(-1, 3) # [N*K, 3]
    
    def plane_radius_sample(self, pc: Tensor):
        T1 = time.time()
        points = pc[:,:3]
        dev = pc.device
        origin = tensor([0, 0, 0], dtype=torch.float32, device=dev) # local frame
        N = points.shape[0]
        
        colors = None
        if pc.shape[1] == 6: # XYZ-RGB
            colors = pc[:,3:]
        
        # Read Configuration
        R = self.cfg.sampling.surface_radius
        K = self.cfg.sampling.nsamples

        # Create  rays and normalize them as normal vectors
        vec_po = origin.reshape(1,3).tile((N,1)) - points # [N,3], vector direction from P to O
        dist = torch.linalg.norm(vec_po, ord=2, dim=-1).reshape((N,1)).tile((1,3)) # [N,3] # ||PO||
        normals = vec_po/dist # [N,3] normal(PO) = PO/||PO||
        d = -torch.sum(normals*points, dim=-1).reshape(N,1).tile((1, K)) #[N,K] -> copy K neighbors

        xp = points[:,0].unsqueeze(-1).tile(1,K) # [N, K]  -> copy K neighbors
        yp = points[:,1].unsqueeze(-1).tile(1,K) 
        zp = points[:,2].unsqueeze(-1).tile(1,K) 
        a = normals[:,0].unsqueeze(-1).tile(1,K) # [N, K]  -> copy K neighbors
        b = normals[:,1].unsqueeze(-1).tile(1,K) 
        c = normals[:,2].unsqueeze(-1).tile(1,K) 
        
        # Sphere center  at p, radius R equation: (x-xp)^2 + (y-yp)^2 + (z-zp)^2 = R^2 (1)
        #                                  (x-xp)^2 + (y-yp)^2 + delta_z^2 - R^2 = 0 (1)
        
        # Plane with normal vector PO via P: PO.(A-P) = 0: [a,b,c].[x-xp, y-yp, z-zp] = 0
        #                                     a.x + b.y + c.z - (a.xp + b.yp + c.zp) = 0
        # Put d = - (a.xp + b.yp + c.zp) =>                      a.x + b.y + c.z + d = 0 (2)
        
        # From (2): x = -(d/a + b.y/a + c.z/a) subtitude into (1): 
        #    (d/a + b.y/a + c.z/a + xp)^2 + (y-yp)^2 + delta_z^2 - R^2 = 0
        # Put: e = (d/a + cz/a + xp) = (d+c.z)/a + xp 
        # =>  (e + b.y/a)^2 + (y-yp)^2 + delta_z^2 - R^2 = 0
        #     e^2 + 2.e.b.y/a + (b/a)^2.y^2 + y^2 - 2.y.yp + yp^2 + delta_z^2 - R^2 = 0
        #     [(b/a)^2 + 1].y^2 + (2.e.b/a - 2.yp).y + (e^2 + yp^2 + delta_z^2 - R^2 ) = 0
        #                 A.y^2       +          B.y   +                C               = 0   
        
        print_var(R, "R")                              
        delta_z = torch.rand((N, K), device=dev)*R*(-2) + R # in (-R, R) size [N,K]
        z = zp + delta_z # [N,K]
        a += 1.0e-15 # [N,K] prevent divided by zeros
        e = (c*z+d)/a + xp # [N,K] 
        
        A = (b/a)**2 + 1 # [N,K] 
        B = (2*b*e)/a - 2*yp # [N,K] 
        C = e**2 +  yp**2 + delta_z**2 - R**2 # [N,K] 
        delta = B**2 - 4*A*C # [N,K] 
        mask = (delta >= 0) # [N, K] of booleans
        self.logger.log(1, f'    Sampling->plane_radius_sample: mask = {mask.shape}, valid = {(mask).sum()}')
               
        choice = torch.randint(1,100, (1,))
        y = torch.zeros_like(z)
        x = torch.zeros_like(z)
        if choice[0] % 2 == 0:
            y[mask] = (-B[mask] - delta[mask]**0.5)/ (2*A[mask]) # or y = (-B + delta**0.5)/ (2*A) are same since Z ~ U(-R,R) [N1]
        else: 
            y[mask] = (-B[mask] + delta[mask]**0.5)/ (2*A[mask])
            
        x[mask] = -(b[mask]*y[mask] + c[mask]*z[mask] + d[mask])/a[mask] #[N1]
        
        
        points_knn = points.reshape(N, 1, 3).tile(1,K,1) # [N,K,3]
        samples_knn = torch.cat([x.reshape(-1,K,1), y.reshape(-1,K,1), z.reshape(-1,K,1)], dim=-1) # [N,K,3]
        
        # Filter singular points
        distances_knn = points_knn.norm(p=2, dim=-1) # [N,K]
        sampled_distances_knn = samples_knn.norm(p=2, dim=-1) # [N,K]
        displacements = sampled_distances_knn - distances_knn # [N,K]
        mask &= (displacements < R) # Add filter with displacements
        
        sampled_points = samples_knn[mask] # [N,K,3] -> [N1, 3]
        
        # print_var(sampled_points, "sampled_points")
        # print_var(displacements, "displacements")
        # exit()
        
        N1 = sampled_points.shape[0]
        shortages = N*(K+1)-N1
        
        samples = torch.cat([points, sampled_points], dim=0) # [N+N1, 3]
        surface_displacements = torch.zeros((N,)).to(points) #[N]
        displacements =  torch.cat([surface_displacements, displacements[mask]], dim=-1) # [N+N1]
        
         # Colors
        # '''
        if colors != None:
            surface_colors = colors.reshape(N, 1, 3).tile(1, K, 1)    # Colors [N,3] -copy-> [N,K,3]
            sampled_colors = surface_colors[mask] # [N1,3]
            sampled_colors = torch.cat([colors, sampled_colors], dim=0) # [N+N1, 3]
            samples = torch.cat([samples, sampled_colors], dim=-1) # [N+N1,6]
        # '''
        
        T2 = time.time()
        self.logger.log(1, f'  Sampling->plane_radius_sample:  expected {N*(K+1)} sampled points,  {N1+N} points are sampled, shortages = {shortages} points, time = {(T2-T1)*1000: 0.2f} ms')
        
         # Calculate OQ = (OQ/OP)*OP
        weights = torch.ones((samples.shape[0],)).to(points) # [N1, K]
        return samples, displacements, weights
    
    def dist_sampling(self, pc: Tensor):
        T1 = time.time()
        points = pc[:,:3] # get XYZ from point cloud
        dev = points.device
        N = points.shape[0]
        
        colors = None
        if pc.shape[1] == 6: # XYZ-RGB
            colors = pc[:,3:]
            
        # Read configurations
        surface_sample_range = self.cfg.sampling.surface_sample_range
        K1 = self.cfg.sampling.surface_nsamples
        K2 = self.cfg.sampling.front_nsamples
        K3 = self.cfg.sampling.back_nsamples
        K = 1 + K1 + K2 + K3
        sigma = self.cfg.sampling.sigma
        front_ratio = self.cfg.sampling.front_ratio # distance from origin OQ: samples from dist*front_ratio, called Q, to surface, there are no points between [0, A]
        distances = torch.norm(points, dim=-1, keepdim=True) # [N,1] distance from P to local original [0,0,0] call OP
        
        # Surface
        surface_displacement = torch.randn(N, K1, device=dev)*surface_sample_range # [N, K1]:  Normal distribution N(0,1)*range: 
        surface_dist_ratio = surface_displacement / distances.tile(1, K1) + 1 # [N,K1]
        
        
        # Front 
        front_delta_ratio = (1.0 - front_ratio) - (sigma*surface_sample_range)/distances.tile(1, K2) # [N,K2] extended ratio from OQ
        front_dist_ratio = torch.rand(N, K2, device=dev)*front_delta_ratio + front_ratio  # [N,K2]: uniform distribution: ratio points in [P, surface]
        front_displacement = (front_dist_ratio - 1.0)*distances.tile(1, K2) # # [N,K2]
        
        # Back
        back_ratio = 1.0 + (sigma*surface_sample_range)/distances.tile(1, K3) # points, call Q, are back surface: extended=(sigma*range)/dist
        back_delta_ratio = (1.0 - sigma*surface_sample_range)/distances.tile(1, K3) # [N,K3] extended ratio from Q
        back_dist_ratio = torch.rand(N, K3, device=dev)*back_delta_ratio + back_ratio  # # [N,K3] ratio points in [Q, surface]
        back_displacement = (back_dist_ratio - 1.0)*distances.tile(1, K3)
        
        surface_sampled_points = surface_dist_ratio.reshape(N, K1,1) .tile(1,1,3)*points.reshape(N, 1, 3).tile(1, K1, 1) #[N,K1,3]
        front_sampled_points = front_dist_ratio.reshape(N, K2,1).tile(1,1,3)*points.reshape(N, 1, 3).tile(1, K2, 1) #[N,K2,3]
        back_sampled_points = back_dist_ratio.reshape(N, K3,1).tile(1,1,3)*points.reshape(N, 1, 3).tile(1, K3, 1) #[N,K3,3]
        
        
        
        
        sampled_points = torch.cat([points.reshape(N, 1, 3), #[N, 1,  3]
                                    surface_sampled_points,  #[N, K1, 3]
                                    front_sampled_points,    #[N, K2, 3]
                                    back_sampled_points],    #[N, K3, 3]
                                     dim=1).reshape(-1,3)    # K=1+K1+K2+K3 -> [N,K,3] -> [N*K,3]
        
        displacements = torch.cat([torch.zeros_like(distances), #[N,1] No displacement for original points, 
                                   surface_displacement,  #[N,K1]
                                   front_displacement,  #[N,K2]
                                   back_displacement], #[N,K3] 
                                  dim=1).reshape(-1)# [N*K] where K=1+K1+K2+K3
        # All sampled distance ratio over given points from origin O : OQ/OP
        sampled_dist_ratio = torch.cat([torch.ones_like(distances), #[N, 1] surface ratio 1:1
                                        surface_dist_ratio, #[N,K1]
                                        front_dist_ratio, #[N,K2]
                                        back_dist_ratio], #[N,K3]
                                        dim=1) # [N, K] where K=1+K1+K2+K3
        
        # Colors
        # '''
        if colors != None:
            surface_colors = colors.reshape(N, 1, 3).tile(1,K1,1)              # Colors [N,3] -> [N,K1,3]
            front_colors = torch.zeros((N, K2, 3)).to(colors) # No color                          [N,K2,3]  
            back_colors = torch.zeros((N, K3, 3)).to(colors)  # No color                          [N,K3,3]  
            sample_colors = torch.cat([colors.reshape(N,1,3), surface_colors, front_colors, back_colors], dim=1) # [N,K,3]
            sample_colors = sample_colors.reshape(-1, 3) # [N*K, 3]
            sampled_points = torch.cat([sampled_points, sample_colors], dim=-1) # [N*K,6]
        # '''
        
        # Calculate OQ = (OQ/OP)*OP
        sampled_distances = (distances.tile(1, K)*sampled_dist_ratio) # [N,K]*[N,K]=[N,K] 
        weights = torch.ones_like(sampled_distances)*(-1) # [N,K]: negative: freespace, positive surface
        scale = self.cfg.sampling.dist_weight_scale # default=0.8
        max_range = self.cfg.pc.max_range # default=60.0
        weights[:, :1+K1] = 1 + (0.5 - distances.tile(1, 1+K1)/max_range)*scale # weight on surface and original points, others = 1
        weights = weights.reshape(-1) # [N,K] --reshape--> [N*K]
        T2 = time.time()
        self.logger.log(1, f'  Sampling->dist_sampling: pc={pc.shape} -> sampled_points={sampled_points.shape} time={(T2-T1)*1000: 0.2f} ms')
        

        ''' Visualization
        step = 10
        from lib.liealgebra import LieAlgebra as la
        T = la.axis_angle_to_transform([3.0, 0.0, 0.0], la.deg2rad(-180))
        frustum = Vis.get_camera_frustum(self.cfg.camera, torch.eye(4)@T.cpu(), depth=1.0)
        surface_sampled_points_with_colors = torch.cat([surface_sampled_points, surface_colors], dim=-1) # [N,K1,6]
        Vis("Samling").draw([surface_sampled_points_with_colors.reshape(-1, pc.shape[1])[::1, :], 
                             front_sampled_points.reshape(-1,3)[::step*2, :], 
                             back_sampled_points.reshape(-1,3)[::step*2, :]], 
                            [Vis.red, Vis.green, Vis.blue], open3d_geometries=[frustum]); exit()
        '''
        
        return sampled_points, displacements, weights
    
    def sample(self, points_torch, normal_torch=None, sem_label_torch=None, color_torch=None):
        """
        Sample training sample points for current scan, get the labels for online training
        input and output are all torch tensors
        points_torch is in the sensor's local coordinate system, not yet transformed to the global system
        """

        # T0 = get_time()

        dev = self.cfg.device
        surface_sample_range = 0.25 # 0.25
        surface_sample_n = 3 # 3
        freespace_behind_sample_n = 1 # 1
        freespace_front_sample_n = 2 # 2
        all_sample_n = (
            surface_sample_n + freespace_behind_sample_n + freespace_front_sample_n + 1
        )  # 1 as the exact measurement. The result is 7: each point add 6 more samples
        free_front_min_ratio = 0.3 # 0.3
        free_sample_end_dist = 1.0 # 1.0

        # get sample points
        # points_torch = points_torch[:100,0:3]
        point_num = points_torch.shape[0]
        distances = torch.linalg.norm( # distance from local frame
            points_torch, dim=1, keepdim=True
        )  # ray distances (scaled) [N]

        # Part 0. the exact measured point
        measured_sample_displacement = torch.zeros_like(distances)
        measured_sample_dist_ratio = torch.ones_like(distances) #[N]

        # Part 1. close-to-surface uniform sampling
        # uniform sample in the close-to-surface range (+- range)
        surface_sample_displacement = (
            torch.randn(point_num * surface_sample_n, 1, device=dev) # uniform distribution over [0,1)
            * surface_sample_range
        ) # [N*3] \in (0, 0.25)

        repeated_dist = distances.repeat(surface_sample_n, 1) # [N*3, 3]
        surface_sample_dist_ratio = (
            surface_sample_displacement / repeated_dist + 1.0
        )  # 1.0 means on the surface: ratio=(r+1) -> point = point*ratio=point(r+1)= point + r*point 
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(
                1, surface_sample_n
            ).transpose(0, 1)
        if color_torch is not None:
            color_channel = color_torch.shape[1]
            surface_color_tensor = color_torch.repeat(surface_sample_n, 1)

        # Part 2. free space (in front of surface) uniform sampling
        # if you want to reconstruct the thin objects (like poles, tree branches) well, you need more freespace samples to have
        # a space carving effect

        sigma_ratio = 2.0
        repeated_dist = distances.repeat(freespace_front_sample_n, 1) # [N*2, 3]
        free_max_ratio = 1.0 - sigma_ratio * surface_sample_range / repeated_dist # 1 - 2*0.25/repeated_dist = 1-0.5/dist
        free_diff_ratio = free_max_ratio - free_front_min_ratio # free_front_min_ratio=0.3, -> 0.7-0.5/dist
        #  = ( 1.0 - sigma_ratio * surface_sample_range / repeated_dist) - free_front_min_ratio
        free_sample_front_dist_ratio = (
            torch.rand(point_num * freespace_front_sample_n, 1, device=dev) 
            * free_diff_ratio # [0, 0.7)
            + free_front_min_ratio # 0.3
        ) # in (0, 1)
        
        free_sample_front_displacement = ( # for SDF label
            free_sample_front_dist_ratio - 1.0 # in (-1, 0) SDF front < 0
            # 1 - free_sample_front_dist_ratio # (0,1)
        ) * repeated_dist
        
         
        if sem_label_torch is not None:
            free_sem_label_front = torch.zeros_like(repeated_dist)
        if color_torch is not None:
            free_color_front = torch.zeros(
                point_num * freespace_front_sample_n, color_channel, device=dev
            )

        # Part 3. free space (behind surface) uniform sampling
        repeated_dist = distances.repeat(freespace_behind_sample_n, 1) # [N, 3]
        free_max_ratio = free_sample_end_dist / repeated_dist + 1.0 # free_sample_end_dist = 1.0 -> ret = 1/dist + 1
        free_behind_min_ratio = 1.0 + sigma_ratio * surface_sample_range / repeated_dist # ret=1+0.5/dist
        free_diff_ratio = free_max_ratio - free_behind_min_ratio # ret = (1/dist + 1) - (1+0.5/dist) = 0.5/dist
        #   = free_sample_end_dist / repeated_dist + 1.0 - (1.0 + sigma_ratio * surface_sample_range / repeated_dist)
        #   = (free_sample_end_dist - sigma_ratio * surface_sample_range )/repeated_dist
        

        free_sample_behind_dist_ratio = (
            torch.rand(point_num * freespace_behind_sample_n, 1, device=dev)
            * free_diff_ratio # in (0, 0.5/dist)
            + free_behind_min_ratio
        )

        free_sample_behind_displacement = (
            free_sample_behind_dist_ratio - 1.0
        ) * repeated_dist
        if sem_label_torch is not None:
            free_sem_label_behind = torch.zeros_like(repeated_dist)
        if color_torch is not None:
            free_color_behind = torch.zeros(
                point_num * freespace_behind_sample_n, color_channel, device=dev
            )

        # T1 = get_time()

        # all together
        all_sample_displacement = torch.cat(
            (
                measured_sample_displacement,
                surface_sample_displacement,
                free_sample_front_displacement,
                free_sample_behind_displacement,
            ),
            0,
        )
        all_sample_dist_ratio = torch.cat(
            (
                measured_sample_dist_ratio,
                surface_sample_dist_ratio,
                free_sample_front_dist_ratio,
                free_sample_behind_dist_ratio,
            ),
            0,
        ) # [N*7,1]
        
        

        repeated_points = points_torch.repeat(all_sample_n, 1) #[N*7,3]
        repeated_dist = distances.repeat(all_sample_n, 1) #[N*7,1]
        all_sample_points = repeated_points * all_sample_dist_ratio #[N*7,3]
        print(f'  DataSampler->sample: all_sample_points={all_sample_points.shape}')

        # depth tensor of all the samples
        depths_tensor = repeated_dist * all_sample_dist_ratio # #[N*7,1] * [N*7,1] 
        
        # get the weight vector as the inverse of sigma
        weight_tensor = torch.ones_like(depths_tensor)

        surface_sample_count = point_num * (surface_sample_n + 1)
        if (
            True # TRUE
        ):  # far away surface samples would have lower weight
            weight_tensor[:surface_sample_count] = (
                1
                + 0.8 * 0.5 # dist_weight_scale=0.8
                - (repeated_dist[:surface_sample_count] / 60.0) # max_range=60
                * 0.8
            )  # [0.6, 1.4]
        # TODO: also try add lower weight for surface samples with large incidence angle

        # behind surface weight drop-off because we have less uncertainty behind the surface
        if False: # FALSE
            dropoff_min = 0.2 * free_sample_end_dist
            dropoff_max = free_sample_end_dist
            dropoff_diff = dropoff_max - dropoff_min
            behind_displacement = all_sample_displacement
            dropoff_weight = (dropoff_max - behind_displacement) / dropoff_diff
            dropoff_weight = torch.clamp(dropoff_weight, min=0.0, max=1.0)
            dropoff_weight = dropoff_weight * 0.8 + 0.2
            weight_tensor = weight_tensor * dropoff_weight


        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        weight_tensor[surface_sample_count:] *= -1.0

        # ray-wise depth
        distances = distances.squeeze(1)

        # assign sdf labels to the samples
        # projective distance as the label: behind +, in-front -
        sdf_label_tensor = all_sample_displacement.squeeze(
            1
        )  # scaled [-1, 1] # as distance (before sigmoid)

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            normal_label_tensor = normal_torch.repeat(all_sample_n, 1)

        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None
        if sem_label_torch is not None:
            sem_label_tensor = torch.cat(
                (
                    sem_label_torch.unsqueeze(-1),
                    surface_sem_label_tensor,
                    free_sem_label_front,
                    free_sem_label_behind,
                ),
                0,
            ).int()

        # assign the color label to the close-to-surface samples
        color_tensor = None
        if color_torch is not None:
            color_tensor = torch.cat(
                (
                    color_torch,
                    surface_color_tensor,
                    free_color_front,
                    free_color_behind,
                ),
                0,
            )

        # T2 = get_time()
        # Convert from the all ray surface + all ray free order to the ray-wise (surface + free) order
        all_sample_points = (
            all_sample_points.reshape(all_sample_n, -1, 3) # [N*7, 1, 3]
            .transpose(0, 1) # [1, N*7,3]
            .reshape(-1, 3)  # [N*7, 3]
        )
        sdf_label_tensor = (
            sdf_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )
        sdf_label_tensor *= -1  # convert to the same sign as

        weight_tensor = (
            weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )
        # depths_tensor = depths_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        if normal_torch is not None:
            normal_label_tensor = (
                normal_label_tensor.reshape(all_sample_n, -1, 3)
                .transpose(0, 1)
                .reshape(-1, 3)
            )
        if sem_label_torch is not None:
            sem_label_tensor = (
                sem_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
            )
        if color_torch is not None:
            color_tensor = (
                color_tensor.reshape(all_sample_n, -1, color_channel)
                .transpose(0, 1)
                .reshape(-1, color_channel)
            )

        return (
            all_sample_points,
            sdf_label_tensor,
            weight_tensor,
        )
        
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

        pc1 = Sampling.sample_plane(tensor([1, 0, 0]).float(), tensor([0, Ly/2, Hz/2]),  Wx, Ly, Hz, npoints)
        pc2 = Sampling.sample_plane(tensor([1, 0, 0]).float(), tensor([-Wx, Ly/2, Hz/2]), Wx, Ly, Hz, npoints)
        pc3 = Sampling.sample_plane(tensor([0, 1, 0]).float(), tensor([Wx/2, 0, Hz/2]), Wx, Ly, Hz, npoints)
        pc4 = Sampling.sample_plane(tensor([0, 1, 0]).float(), tensor([Wx/2, -Ly, Hz/2]), Wx, Ly, Hz, npoints)
        pc5 = Sampling.sample_plane(tensor([0, 0, 1]).float(), tensor([Wx/2, Ly/2, 0]), Wx, Ly, Hz, npoints)
        pc6 = Sampling.sample_plane(tensor([0, 0, 1]).float(), tensor([Wx/2, Ly/2, -Hz]), Wx, Ly, Hz, N-npoints*5)

        pc =  torch.cat([pc1, pc2, pc3, pc4, pc5, pc6], dim=0)
        edge_index = tensor([[0,0,0, 1,1,1, 2,2,2, 3,3,3, 4,4,4, 5,5,5, 6,6,6, 7,7,7],
                                    [1,3,4, 0,2,5, 1,3,7, 0,2,6, 0,5,6, 1,4,7, 3,4,7, 2,5,6]]).long()
        N = pc.shape[0]
        sdf = torch.zeros((N,)).to(pc)
        return pc, keypoints, edge_index, sdf
    
    def sample_sphere(r=1, N=2000):# Reference: https://mathworld.wolfram.com/SpherePointPicking.html
        theta = torch.FloatTensor(N).uniform_(0, 2*torch.pi)
        u = torch.FloatTensor(N).uniform_(-1, 1)
        x = torch.sqrt(1-u**2)*torch.cos(theta)*r
        y = torch.sqrt(1-u**2)*torch.sin(theta)*r
        z = u*r
        pc = torch.cat([x[:,None], y[:,None], z[:,None]], axis=1)

        #Generate keypoints
        Nc=8 # number of points each circle
        n=3*Nc+2 # Total key points
        keypoints = torch.zeros(n, 3)
        keypoints[0,:] = torch.Tensor([math.sqrt(1-(-1)**2)*1*r, math.sqrt(1-(-1)**2)*0*r, -r]) #bottom
        keypoints[n-1,:] = torch.Tensor([math.sqrt(1-(1)**2)*1*r, math.sqrt(1-(1)**2)*0*r, r]) #top
        step_size =  2*torch.pi/Nc
        theta = torch.linspace(0, step_size*(Nc-1), Nc)
        print(theta)
        def func(u_val):
            u = torch.Tensor([u_val])
            x = torch.sqrt(1-u**2)*torch.cos(theta)*r
            y = torch.sqrt(1-u**2)*torch.sin(theta)*r
            z = u.repeat(Nc)*r
            return torch.cat([x[:,None], y[:,None], z[:,None]], dim=1)
        keypoints[0*Nc+1:(Nc*1 + 1),:] = func(-0.5) #1
        keypoints[1*Nc+1:(Nc*2 + 1),:] = func(0)
        keypoints[2*Nc+1:(Nc*3 + 1),:] = func(0.5)

        #Generate edges
        layers = torch.zeros(3, Nc)
        for i in range(3):
            for j in range(Nc):
                layers[i, j] = 1 + i*Nc + j
            # print(f'Layer {i} : {layers[i]}')
        # Horizontal lines
        edges = []
        ind_start, ind_end = 0, Nc-1
        for i in range(3):
            for j in range(Nc):
                if j == ind_start:
                    edges.append([layers[i, j], layers[i, ind_end]])
                    edges.append([layers[i, j], layers[i, j+1]])
                elif j == ind_end:
                    edges.append([layers[i, j], layers[i, j-1]])
                    edges.append([layers[i, j], layers[i, ind_start]])
                else:
                    edges.append([layers[i, j], layers[i, j-1]])
                    edges.append([layers[i, j], layers[i, j+1]])
        #Vertical
        for j in range(Nc):
            edges.append([0, layers[0, j]]) #bottom
            edges.append([n-1, layers[2, j]]) #top
        #connection between layers
        for j in range(Nc):
            edges.append([layers[1, j], layers[0, j]]) # Connetion betweent layer 1 & 0
            edges.append([layers[1, j], layers[2, j]]) # Connetion betweent layer 1 & 2

        edge_index = torch.Tensor(edges).long().T
        # print(edge_index)
        return pc, keypoints, edge_index
