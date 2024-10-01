import os 
import torch
from torch import nn


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def mcc_loss(x, reweight=False, dim=2, class_num=32):
    p = x.softmax(dim) # (B, L, D)
    if reweight: # (B, L, D) * (B, L, 1) 
        target_entropy_weight = softmax_entropy(x, dim=2).detach() # instance-wise entropy (B, L, D)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (B, L)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = torch.matmul(p.mul(target_entropy_weight.unsqueeze(-1)).transpose(2, 1), p)
    else:    
        cov_matrix_t = p.transpose(2, 1).mm(p) # (B, D, L) * (B, L, D) -> (B, D, D)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=-1, keepdim=True)
    tr = 0
    for i in range(x.shape[0]):
        tr += torch.trace(cov_matrix_t[i])
    mcc_loss = (torch.sum(cov_matrix_t) - tr) / class_num / x.shape[0]
   
    return mcc_loss

def div_loss(x, non_blank=None, L_thd=64):
    # maximize entropy of class prediction for every time-step in a utterance 
    # x (1, L, D)
    loss = 0
    x = x.squeeze(0)
    L = x.shape[0]

    if non_blank is not None: 
        cls_pred = x.mean(0)[1:] # (D, )
    else:
        cls_pred = x.mean(0) # (D, )

    loss = -softmax_entropy(cls_pred, 0)

    return loss
