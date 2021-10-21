import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        log_preds = F.log_softmax(output, dim=-1)
        loss = -log_preds.mean()
        return loss * self.eps + (1-self.eps) * F.nll_loss(log_preds, target)