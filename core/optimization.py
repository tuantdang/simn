from torch import Tensor, diag
from torch.linalg import inv
import torch
from lib.utils import print_var, detect_nan
from lib.liealgebra import LieAlgebra as la

'''
# p in se(3)              -> T SE(3)
#   R^6                   -> R^(4x4) 
#[ax, ay, az, tx, ty, tz] -> R^(4x4): where [ax, ay, az] axis angle reprsentation
#Levenberg-Marquartd: 
#    (J^T.J + lamda.I).delta_p = -J^T err(p) or  
# => delta_p = -(J^T.J + lamda.I)^-1.(J^T.err(p))
'''
def Levenberg_Marquardt(
        pc: Tensor, #[N, input_dim]
        gradients: Tensor, # [N, 3] gradients
        y_hat: Tensor,  #[N]  prediction
        y_truth: Tensor, #[N] grouth truth
        w: Tensor, #[N] weights for points
        ld: float=1.0e-4 # Levenberg-Marquart lambda
    ):
    xyz = pc[:,:3]
    xyz_grad = gradients[:,:3]
    I = torch.eye(6, device=pc.device) # Identity matrix
    w = w.unsqueeze(-1) # [N, 1]
    err = (y_hat-y_truth).unsqueeze(-1) # [N, 1]
    delta_rotation = torch.cross(xyz, xyz_grad, dim=-1) # [N,3] cross [N,3] = [N,3]: right-hand law
    J = torch.cat([delta_rotation, xyz_grad], dim=-1) # [N,3] concate [N, 3] = [N,6]
    H = J.T@(w*J) # [6,N]@[N,6]=[6,6] Hessian matrix approximate the second order (curveture) of cost function (error function)
  
    #Levenberg-Marquartd: (J^T.J + lamda.I).delta_p = -J^T err(p) or  => delta_p = -(J^T.J + lamda.I)^-1.(J^T.err(p))
    # I = diag(diag(H))
    delta_p = -inv(H + ld*I)@((w*J).T@err) # [6,1]
    
    # se(3) -> SE(3)
    delta_p = delta_p.squeeze(-1) # [6]
    delta_T = la.se3_exp_map(delta_p) # R^6 -> R^(4x4) or se(3) -> SE(3)
    
    # detect_nan(delta_T, 'Levenberg_Marquardt: se3_exp_map')
    # mse = (w*err.squeeze(-1)**2).mean() # mean square error sum(y-y^)**2: err[N,1] -> [N] ->mse=[1]
    cov = inv(H + ld*I) # Cov(p)= H^-1(p) # MLE: Maxium Likelihood Estimation with respect to parameter p in R^6
    return delta_T, cov