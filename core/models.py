# Author:  Tuan Dang   
# Email:   tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2024 Tuan Dang, all rights reserved


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lib.utils import print_var
from lib.logger import Logger

class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        input_dim = cfg.feature.feature_dim +  cfg.feature.input_dim # 8 + 3 = 11
        hidden_dim = cfg.decoder.hidden_dim
        hidden_level = cfg.decoder.hidden_level
        output_dim = cfg.decoder.output_dim
        
        layers = []
        for i in range(hidden_level):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim, cfg.decoder.bias_on))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, cfg.decoder.bias_on))
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(hidden_dim, output_dim, cfg.decoder.bias_on)

        self.sdf_scale = 1.0
        if cfg.loss.type == "bce":
            self.sdf_scale = cfg.loss.logistic_gaussian_ratio * cfg.loss.sigma_sigmoid_m # 0.55*0.1 = 0.055

        self.to(cfg.device)

    def mlp(self, features: Tensor, #[N,K]
            ):
        use_leaky_relu = self.cfg.decoder.use_leaky_relu
        for k, l in enumerate(self.layers):
            if k == 0:
                if use_leaky_relu: # used
                    h = F.leaky_relu(l(features)) # [N,F] -> [N,hidden_dim]
                else:
                    h = F.relu(l(features))
            else:
                if use_leaky_relu: # used
                    h = F.leaky_relu(l(h)) # # [N,hidden_dim] -> [N,hidden_dim]
                else:
                    h = F.relu(l(h))
        out = self.lout(h) # [N,hidden_dim] -> [N,out_dim]
        return out

   
    def sdf(self, features: Tensor, # [N,F] -> [N]
            ): # return [N]
        out = self.mlp(features).squeeze(1) * self.sdf_scale # [N,out_dim=1] -> [N]
        return out
    
    
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PCDTransformer(nn.Module):
    def __init__(self, cfg=None):
        super(PCDTransformer, self).__init__()
        d_model=64 
        nhead=4
        num_encoder_layers=1
        dim_feedforward=128
        dropout=0.1
        
        input_dim = cfg.feature.feature_dim +  cfg.feature.input_dim # 8 + 3 = 11
        self.input_layer = nn.Linear(input_dim, d_model)  # Embed 3D points to d_model
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = TransformerEncoder(encoder_layer, num_encoder_layers)
        self.sdf_predictor = nn.Linear(d_model, 1)  # Predict a single SDF value
        self.to(cfg.device)
        
    def sdf(self, features: Tensor #[N,K] 
            ):
        point_features = self.input_layer(features)  # (N, d_model)
        encoded_features = self.transformer(point_features)  # (N + M, d_model)
        sdf_values = self.sdf_predictor(encoded_features)  # (N, 1)
        
        return sdf_values.squeeze(-1) # [N]