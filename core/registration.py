import torch
from torch import Tensor
from rich import print
from core.latentfeature import LatentFeature
import math
from lib.utils import print_var, voxel_down_sample, detect_nan
from core.models import Decoder
from lib.logger import Logger
from core.optimization import Levenberg_Marquardt
from lib.liealgebra import LieAlgebra as la
from torch.autograd import grad
from torch.linalg import inv
from core.pgo import PoseGraphOptimization
import numpy as np

class Registration:
    def __init__(self, cfg, latentfeature:LatentFeature=None, decoder:Decoder=None, logger:Logger=None):
        self.cfg = cfg
        self.latentfeature = latentfeature
        self.decoder = decoder
        self.logger = logger
        self.frame_id = 0
        
        # Odometry managements
        self.poses = torch.eye(4, device=cfg.device).view(-1, 4, 4) # First pose at T0
        self.odometry =  torch.empty((0, 4, 4), device=cfg.device, dtype=cfg.dtype.point) # o1   = relative_pose(T0, T1)..
        self.cov_poses = torch.empty((0, 6, 6), device=cfg.device, dtype=cfg.dtype.point) # cov1 =
        self.local_frame_id = []
        
        # First local frame
        self.local_frame_id.append(0)
        
    def register(self, pc:Tensor, #pc: [N,3]
                frame_id=1, #register at frame id = 1
                from_id:int=-1, # with repsect to data from_id to to_id
                to_id:int=-1): 
        
        self.latentfeature.mode = 2
        self.frame_id = frame_id
        
        # Read cofiguration
        voxel_size = self.cfg.pc.vs_reg # 0.6
        min_grad_magnitude = self.cfg.tracking.min_grad_magnitude
        max_grad_magnitude = self.cfg.tracking.max_grad_magnitude
        GM_dist = self.cfg.tracking.GM_dist
        GM_grad = self.cfg.tracking.GM_grad
        n_iter = self.cfg.tracking.n_iter
        
        reg_points, _ = voxel_down_sample(pc, voxel_size)
        
        # if (from_id >= 0 and to_id >= 0) and (from_id < to_id and to_id <= frame_id): # uses timestamp data 
        #     T = self.poses[to_id]
        #     self.logger.log(1, f'  Registration->register: reg_points={reg_points.shape}, translation={T[:3,3]} with loop detection')
        # else:
        T = self.poses[-1,...] # Predict next pose as last pose
        self.logger.log(1, f'  Registration->register: reg_points={reg_points.shape}, translation={T[:3,3]}')
        
        # n_iter = 2
        last_mean_err = 1.0e5
        for iter in range(n_iter):
            points = reg_points.clone()
            points[:,:3] = la.transform(reg_points[:,:3], T) # apply transform T to registration points
            sdf_pred, sdf_grad, mask = self.query_points(points, from_id, to_id) # sdf_pred: [N], sdf_grad: [N,3], mask: [N]
            
            # Assume all surface points
            sdf_values = torch.zeros(points.shape[0], device=points.device) # [N] of zeros: all surfaces points
            
            grad_magnitude = sdf_grad.norm(dim=-1) # [N] distance when p=2: d=||grad||
            # grad_unit = sdf_grad / grad_magnitude.unsqueeze(-1) # [N,3]/[N,1] = [N,3]
            valid_mask = mask & (min_grad_magnitude < grad_magnitude) & (grad_magnitude < max_grad_magnitude)
            
            # Valid data
            valid_points = points[valid_mask]
            valid_sdf_grad = sdf_grad[valid_mask] #[N1]
            valid_grad_magnitude = grad_magnitude[valid_mask] # [N1] where N1 < N
            valid_sdf_pred = sdf_pred[valid_mask] #[N1]
            valid_sdf_values = sdf_values[valid_mask] #[N1]
            err = valid_sdf_pred - valid_sdf_values #[N1]
            
            mean_err = torch.abs(err).mean()
            if (mean_err - last_mean_err)/last_mean_err > 1.0: # error get more than double
                last_mean_err = mean_err
                self.logger.log(1, f'  Registration->register: too much error!')
                exit()
            
            # Weight calculation
            grad_anomaly = (valid_grad_magnitude - 1.0)
            w_grad = (GM_grad/(GM_grad +  grad_anomaly**2))**2 # GM gradident weighting
            w_err = (GM_dist/(GM_dist +  err**2))**2 # GM error  weighting
            w = w_grad*w_err
            w = w/(2.0*w.mean()) # normalize weight
            lamda_factor = self.cfg.tracking.lm_lambda
            delta_T, cov = Levenberg_Marquardt(pc=valid_points, gradients=valid_sdf_grad, 
                                        y_hat=valid_sdf_pred, y_truth=valid_sdf_values, 
                                        w=w, ld=lamda_factor)
            
            diag = torch.diag(cov).cpu().numpy() # 6D covariances
            self.logger.write_cov(f'{frame_id:03d}, {iter:03d}, {diag[0]:+0.4f}, {diag[1]:+0.4f}, {diag[2]:+0.4f}, {diag[3]:+0.4f}, {diag[4]:+0.4f}, {diag[5]:+0.4f}')
            
            self.logger.write_weight(f'{frame_id:03d}, {iter:03d}, {mask.sum():04d}, {valid_mask.sum():04d}, {w_grad.mean():0.3f}, {w_err.mean():0.3f}')
            
          
            axis_angle = la.so3_log_map(delta_T[:3,:3]) # rotation matrix to axis_angle
            _, angle_rad = la.extract_axis_angle(axis_angle)
            angle_deg = la.rad2deg(angle_rad)
            trans = delta_T[0:3,3].norm(p=2)
            
            if (trans < self.cfg.tracking.translation_thres # 1.0e-3=0.001
                and abs(angle_deg) < self.cfg.tracking.rotation_thres): # 1.0e-2
                # print(f'    > {angle_deg}<>{self.cfg.tracking.translation_thres}, {trans} <> {self.cfg.tracking.translation_thres}')
                break
            T = delta_T@T 
            
            if self.cfg.verbose >= 1: # Logging 
                # Gradient
                gm=valid_sdf_grad.mean(dim=0)
                self.logger.write_sdf(f'{mask.sum():04d}, {valid_mask.sum():04d}, {valid_sdf_pred.mean():0.4f}, {valid_grad_magnitude.mean():0.4f}, {gm[0]:0.4f}, {gm[1]:0.4f}, {gm[2]:0.4f}')
        
                # Pose
                txt = f'{frame_id:03d}, {iter:03d}, {mask.sum():04d}, {valid_mask.sum():04d}, '
                txt += la.view_transform(delta_T)
                # txt += ", " + la.view_transform(T)
                self.logger.write_pose(txt)
            
        
        odom = inv(self.poses[-1])@T # odom = Ti^-1@Ti+1
        
        self.odometry = torch.cat([self.odometry, odom.unsqueeze(0)], dim=0) # add odom between (id-1, id)
        self.poses = torch.cat([self.poses, T.unsqueeze(0)], dim=0) # add new pose at id
        self.cov_poses = torch.cat([self.cov_poses, cov.unsqueeze(0)], dim=0)
        return T
    
    def query_points(self, query:Tensor, # [N,3]
                        from_id:int=-1, # with repsect to data from_id to to_id
                        to_id:int=-1,):  
       
        knn = self.cfg.query.knn
        N = query.shape[0]
        sdf_pred = torch.zeros(N).to(query)    # [N]    
        sdf_grad = torch.zeros(N, self.cfg.feature.input_dim).to(query) # [N, input_dim] gradient for each dimension
        
        mask = torch.full((N,), False, dtype=torch.bool, device=self.cfg.device) # [N] mask valid points with full knn
        bs = self. cfg.batchsize.infer
        n_iter = math.ceil(N/bs)
        self.logger.log(2, f'  Registration->query_points: points={query.shape}, bs={bs}, n_iter={n_iter}')
        
        for iter in range(n_iter):
            head = iter*bs
            tail = min(N, (iter+1)*bs)
            batch_points = query[head:tail, :].clone()
            # detect_nan(batch_points, 'Query points')
            with torch.autograd.set_detect_anomaly(True):
                batch_points.requires_grad_(True) # [N,F]
                (batch_features, _, # batch_features = [N,F+input_dim]
                 batch_knn_count # batch_knn_count = [N,knn]
                )= self.latentfeature.query_features(batch_points, False, from_id, to_id)
                batch_sdf = self.decoder.sdf(batch_features) # [N]
                grad_weight_out = torch.ones_like(batch_sdf, requires_grad=False).to(batch_sdf) # [N] weight on each point
                batch_sdf_grad = grad(inputs=batch_points, outputs=batch_sdf, grad_outputs=grad_weight_out, 
                                                        create_graph=True, retain_graph=True, only_inputs=True)[0] # [N,3]
            
                self.logger.log(2, f'    > batch_points={batch_points.shape}, batch_sdf={batch_sdf.shape}, batch_sdf_grad={batch_sdf_grad.mean(dim=0)}')
                
                sdf_pred[head:tail] = batch_sdf.detach()
                sdf_grad[head:tail, :] = batch_sdf_grad.detach()
                mask[head:tail] = (batch_knn_count >= knn) # only point having more than k neighbors are valid
                self.logger.log(2, f'    > Number valid query points {mask.sum()}/{N}')
        return sdf_pred, sdf_grad, mask
    
    
    def detect_new_local_frame(self
                               ):
        start_fid = self.local_frame_id[-1] # Current Local Frame
        cur_pose = self.poses[-1] # [1, 4,4]
        batch_prev_poses = self.poses[start_fid:-1] # [N, 4, 4] from.. to last-1
        bs = batch_prev_poses.shape[0] # N previous poses
        batch_cur_pose = cur_pose.tile(bs,1,1) # [1,4,4] -> [N,4,4]
        odom_cur2prev = torch.bmm(torch.linalg.inv(batch_prev_poses), batch_cur_pose)    # Odom from current pose to previous poses in batch
        
        # Accumulated Translations
        odom_trans = self.odometry[start_fid:,:3,3] # [N, 3]
        trans_btw_frames = odom_trans.norm(p=2, dim=-1)
        acc_trans = torch.sum(trans_btw_frames, dim=-1)
        
        batch_axis_angle = la.so3_log_map_batch(odom_cur2prev[:,:3,:3]) # [bs, 3]
        max_angle_values, _ = torch.max(torch.abs(batch_axis_angle), dim=0) # [3]
        angle_x, angle_y, angle_z = la.rad2deg(max_angle_values)
        if angle_x > self.cfg.local_frame.max_angle_x or angle_y > self.cfg.local_frame.max_angle_y or angle_z > self.cfg.local_frame.max_angle_z:
            # print(f'    > frame_id = {self.frame_id}: {angle_x}, {angle_y}, {angle_z}, acc_trans = {acc_trans}')
            self.local_frame_id.append(self.frame_id)
            self.logger.write_local_frameid(f'{self.frame_id:05d}, {angle_x:02.3f}, {angle_y:02.3f}, {angle_z:02.3f}, {acc_trans:02.6f}, {self.frame_id-start_fid:03d}')
       
        self.detect_loop_closure()
       
    def detect_loop_closure(self):
        #Read configuration
        min_loop_dist = self.cfg.local_frame.min_loop_dist
        min_nframes_btw_cur_start = self.cfg.local_frame.min_nframes_btw_cur_start
        
        cur_fid = self.frame_id 
        start_fid = self.local_frame_id[-1]
        cur_pose = self.poses[-1]
        min_val = -1.0
        loop_id = -1
        if cur_fid - start_fid > min_nframes_btw_cur_start:
            n = cur_fid - (start_fid + min_nframes_btw_cur_start)
            prev_pos = self.poses[start_fid:start_fid+n, :3, 3] # [n,3]
            cur_pos = cur_pose[:3,3].reshape(1,3) #[1,3]
            distances = torch.cdist(cur_pos, prev_pos).reshape(-1)
            min_val, min_id = torch.min(distances, dim=-1)
            min_val = min_val.item()
            # self.logger.write_loop(f'{cur_fid:05d}, {start_fid:05d}, {loop_id:05d}, {min_val.item():0.4f}')
            if min_val < min_loop_dist:
                loop_id = start_fid + min_id
                odom = torch.linalg.inv(self.poses[loop_id])@cur_pose
                print(f'  Registration->detect_loop_closure: min_val={min_val}, start_fid={start_fid}, loop_id={loop_id}, cur_fid={cur_fid}')
                # Build PGO and optimize pose here
                # return loop_id, odom
                pgo = PoseGraphOptimization()
                pgo.add_prior(start_fid, 
                              self.poses[start_fid].cpu().numpy(), 
                              self.cov_poses[start_fid].cpu().numpy())
                # pgo.add_init(start_fid, init_pose=self.poses[start_fid].cpu().numpy()) # Better guess
                pgo.add_init(start_fid, init_pose=np.eye(4)) # First node guess as Identity matrix (debugging)
                for id in range(start_fid, cur_fid): # [start_dif..cur_id-1]
                    pgo.add_odom(id, id+1, self.odometry[id].cpu().numpy(), 
                                 self.cov_poses[id].cpu().numpy())
                    pgo.add_init(id+1, self.poses[id+1].cpu().numpy())
                # Add loop closure
                pgo.add_odom(loop_id, cur_fid, odom.cpu().numpy())
                est_poses = pgo.optimize()
                for i, pose in  enumerate(est_poses):
                    before = self.poses[start_fid+i].cpu().numpy()[:3,3]
                    after = pose[:3,3]
                    diff = after - before
                    # print(f'    > PGO: diff={diff}')
    
        self.logger.write_loop(f'{cur_fid:05d}, {start_fid:05d}, {loop_id:05d}, {min_val:0.4f}')
        return None, None