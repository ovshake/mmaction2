# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import torch.distributed as dist

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class MocoLoss(BaseWeightedLoss):

    def __init__(self, temperature=0.5, loss_weight=1.0):
        super().__init__(loss_weight=loss_weight) 
        self.temperature = temperature 


    def _forward(self, q, k, queue):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.temperature
        l_pos /= self.temperature 
        l_pos = l_pos.exp().sum() 
        logits = logits.exp().sum() 
        loss =  - torch.log ( l_pos/ (logits + 1e-8) )
        losses = {'moco_loss': self.loss_weight * loss} 
        return losses 
