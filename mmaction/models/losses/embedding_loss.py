import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from ..builder import LOSSES
from .base import BaseWeightedLoss
from torch.nn.functional import normalize


@LOSSES.register_module()
class EmbeddingLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.mse_loss = nn.MSELoss()

    def _forward(self, teacher_embeddings, student_embeddings):
        assert len(teacher_embeddings) == len(student_embeddings)
        # import ipdb; ipdb.set_trace()
        loss_mse = self.mse_loss(teacher_embeddings, student_embeddings)
        return {"loss_embedding": loss_mse}



@LOSSES.register_module()
class KD_Loss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.kl_div = F.kl_div()
        self.temperature = temperature

    def _forward(self, teacher_embeddings, student_embeddings):
        assert len(teacher_embeddings) == len(student_embeddings)
        # import ipdb; ipdb.set_trace()
        kd_loss = self.kl_div(teacher_embeddings, student_embeddings,reduction="none").sum(1).mean()
        kd_loss*=self.temperature**2
        return {"loss_embedding": kd_loss}
