import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.size(0) != targets.size(0):
            raise ValueError("Shape mismatch between inputs and targets")
        p = F.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = self.alpha * ((1 - p_t) ** self.gamma)
        log_p_t = torch.log(p_t + 1e-8)
        loss = -focal_weight * log_p_t
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
