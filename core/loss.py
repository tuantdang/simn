# @author    Tuan Dang   
# Email: tuan.dang@uta.edu, dangthanhtuanit@gmail.com
# Copyright (c) 2025 Tuan Dang, all rights reserved

from torch import nn, Tensor
import torch
from lib.utils import print_var

'''
BCEWithLogitsLoss: loss(x,y) = -[y*log(sigmod(x)) + (1-y)*log(1-sigmod(x))], where y=0 or 1
'''
def sdf_bce_loss(pred:Tensor, # [N]
                 label:Tensor, # [N]
                 weights:Tensor=None, # [N] 
                 sigma=0.055,  bce_reduction="mean"):
    if weights is not None:
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction, weight=weights) # loss with sigmod activation
    else: # used this
        loss_bce = nn.BCEWithLogitsLoss(reduction=bce_reduction) # loss with sigmod activation
    label_op = torch.sigmoid(label/sigma) # change label to [0..1]
    loss = loss_bce(pred/sigma, label_op)
    return loss

# Step 1: get previous parameters and compute fisher information matrix before training
#         prev_params = {name: param.clone().detach() for name, param in model.named_parameters()}
#         fisher = compute_fisher(model, prev_observation, loss)
# Step 2: Compute current loss with current observations
#          outputs = model(new_observation)
#          loss = criterion(outputs, gt)
# Step 3: Update EWC loss: loss += ewc_loss(model, fisher, prev_params, lambda_ewc)
# Step 4: Update model with loss

def compute_fisher(model, data_loader, criterion):
    fisher = {name: torch.zeros(param.size()) for name, param in model.named_parameters() if param.requires_grad}
    model.eval()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(model.device), labels.to(model.device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.pow(2).detach()
    fisher = {name: f / len(data_loader) for name, f in fisher.items()}
    return fisher

def ewc_loss(model, fisher, prev_params, lambda_ewc):
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher:
            loss += (fisher[name] * (param - prev_params[name]).pow(2)).sum()
    return lambda_ewc * loss


