# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from ..builder import LOSSES
from .base import BaseWeightedLoss
from torch.nn.functional import normalize


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather[dist.get_rank()] = tensor

    output = torch.cat(tensors_gather, dim=0)
    return output

@LOSSES.register_module()
class SlowFastSelfSupervisedLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature

    def _calculate_cosine_similarity(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


    def _forward(self, slow_features, fast_features):
        batch_size = slow_features.shape[0]
        similarity = self._calculate_cosine_similarity(slow_features, fast_features)
        similarity = similarity / self.temperature
        similarity = similarity.exp()
        mask = torch.eye(batch_size, dtype=torch.bool)
        positives = similarity[mask].sum(axis=-1)
        negatives = similarity[~mask].sum(axis=-1)
        loss = - torch.log(positives / (positives + negatives + 1e-8))
        ret_dict = {'slowfast_selfsupervised_loss': self.loss_weight * loss}
        return ret_dict


@LOSSES.register_module()
class ContrastiveLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5, name=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.name = name

    def _calculate_cosine_similarity(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


    def _forward(self, slow_features, fast_features):
        if dist.is_initialized():
            slow_features = concat_all_gather(slow_features)
            fast_features = concat_all_gather(fast_features)
        batch_size = slow_features.shape[0]
        similarity = self._calculate_cosine_similarity(slow_features, fast_features)
        similarity = similarity / self.temperature
        similarity = similarity.exp()
        mask = torch.eye(batch_size, dtype=torch.bool)
        positives = similarity[mask].sum(axis=-1)
        negatives = similarity[~mask].sum(axis=-1)
        loss = - torch.log(positives / (positives + negatives + 1e-8))
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else:
            ret_dict = {f'{self.name}_contrastive_loss': loss}
        return ret_dict

@LOSSES.register_module()
class SingleInstanceContrastiveLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5, name=None):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.name = name

    def _calculate_cosine_similarity(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


    def _forward(self, features_a, features_b):
        if dist.is_initialized():
            slow_features = concat_all_gather(features_a)
            fast_features = concat_all_gather(features_b)
        similarity = self._calculate_cosine_similarity(features_a, features_b)
        similarity = similarity / self.temperature
        similarity = similarity.exp()
        # Isolating the diagonal elements because we expect the positive
        # elements to be in the diagonals
        diag_elems = torch.diagonal(similarity, 0)
        row_sum = similarity.sum(0)  # Taking sum across row
        col_sum = similarity.sum(1)  # Taking sum across columns
        loss = -(
            torch.log(diag_elems / row_sum + 1e-8).sum()
            + torch.log(diag_elems / col_sum + 1e-8).sum()
        )
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else:
            ret_dict = {f'{self.name}_contrastive_loss': loss}
        return ret_dict



@LOSSES.register_module()#BaseWeightedLoss
class SingleInstanceContrastiveLossv2(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5, name=None,
                use_row_sum_a=False,
                use_row_sum_b=False,
                use_positives_in_denominator=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.name = name
        self.use_row_sum_a = use_row_sum_a
        self.use_row_sum_b = use_row_sum_b
        self.use_positives_in_denominator = use_positives_in_denominator

    def _calculate_cosine_similarity(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


    def _forward(self, features_a, features_b):
        if dist.is_initialized():
            features_a = concat_all_gather(features_a)
            features_b = concat_all_gather(features_b)
        batch_size = features_a.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool)
        cross_similarity = self._calculate_cosine_similarity(features_a, features_b)
        a_similarity = self._calculate_cosine_similarity(features_a, features_a)
        # print(type(a_similarity))
        # print(a_similarity)
        a_similarity[mask] = 0.
        b_similarity = self._calculate_cosine_similarity(features_b, features_b)
        b_similarity[mask] = 0.

        cross_similarity = cross_similarity / self.temperature
        cross_similarity = cross_similarity.exp()


        a_similarity = a_similarity / self.temperature
        a_similarity = a_similarity.exp()

        b_similarity = b_similarity / self.temperature
        b_similarity = b_similarity.exp()

        # Isolating the diagonal elements because we expect the positive
        # elements to be in the diagonals
        diag_elems = torch.diagonal(cross_similarity, 0)

        row_sum_cross = cross_similarity.sum(0)  # Taking sum across row
        row_sum_a = a_similarity.sum(0)
        row_sum_b = b_similarity.sum(0)

        # We are taking
        denominator = row_sum_cross
        if self.use_row_sum_a:
            denominator += row_sum_a
        if self.use_row_sum_b:
            denominator += row_sum_b
        if self.use_positives_in_denominator:
            denominator += diag_elems

        denominator += 1e-8

        loss = - torch.log(diag_elems / denominator).mean()
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else:
            ret_dict = {f'{self.name}_contrastive_loss': loss}
        return ret_dict

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


@LOSSES.register_module()
class SingleInstanceContrastiveLossv2_moco_t(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.2, name=None, # other contrastiv method temperature=0.07
                use_row_sum_a=False,
                use_row_sum_b=False,
                use_positives_in_denominator=False):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.name = name
        self.use_row_sum_a = use_row_sum_a
        self.use_row_sum_b = use_row_sum_b
        self.use_positives_in_denominator = use_positives_in_denominator

    def _calculate_cosine_similarity(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt


    def _forward(self, features_a, features_b):
        if dist.is_initialized():
            features_a = concat_all_gather(features_a)
            features_b = concat_all_gather(features_b)
        batch_size = features_a.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool)
        cross_similarity = self._calculate_cosine_similarity(features_a, features_b)
        a_similarity = self._calculate_cosine_similarity(features_a, features_a)
        # print(type(a_similarity))
        # print(a_similarity)
        a_similarity[mask] = 0.
        b_similarity = self._calculate_cosine_similarity(features_b, features_b)
        b_similarity[mask] = 0.

        cross_similarity = cross_similarity / self.temperature
        cross_similarity = cross_similarity.exp()


        a_similarity = a_similarity / self.temperature
        a_similarity = a_similarity.exp()

        b_similarity = b_similarity / self.temperature
        b_similarity = b_similarity.exp()

        # Isolating the diagonal elements because we expect the positive
        # elements to be in the diagonals
        diag_elems = torch.diagonal(cross_similarity, 0)

        row_sum_cross = cross_similarity.sum(0)  # Taking sum across row
        row_sum_a = a_similarity.sum(0)
        row_sum_b = b_similarity.sum(0)

        # We are taking
        denominator = row_sum_cross
        if self.use_row_sum_a:
            denominator += row_sum_a
        if self.use_row_sum_b:
            denominator += row_sum_b
        if self.use_positives_in_denominator:
            denominator += diag_elems

        denominator += 1e-8

        loss = - torch.log(diag_elems / denominator).mean()
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else:
            ret_dict = {f'{self.name}_contrastive_loss': loss}
        return ret_dict

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@LOSSES.register_module()
class SimSiamCosineSimLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, name=None,):
        super().__init__()
        self.loss_weight = loss_weight
        self.name = name

    def D(self, p, z):
        p = normalize(p, dim=1)
        z = normalize(z, dim=1)
        return  - (p * z).sum(dim=1).mean() /2

    def _forward(self, features_a, features_b, number):
        loss = self.D(features_a, features_b.detach())
        if not self.name:
            ret_dict = {'cossim_loss': loss}
        else:
            ret_dict = {f'{number}_{self.name}_cossim_loss': loss}
        return ret_dict


@LOSSES.register_module()
class SimSiamLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, name=None,):
        super().__init__()
        self.loss_weight = loss_weight
        self.name = name
        self.criterion_L = nn.CosineSimilarity(dim=1).cuda()#????



    def _forward(self, p1, p2, z1, z2 ):
        loss = -(self.criterion_L(p1, z2).mean() + self.criterion_L(p2, z1).mean()) * 0.5
        if not self.name:
            ret_dict = {'cossim_loss': loss}
        else:
            ret_dict = {f'{self.name}_cossim_loss': loss}
        return ret_dict




