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
class Symmetric_ContrastiveLossv2_div(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5, name=None, # other contrastiv method temperature=0.07
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


    def _forward(self, p1, p2, z1, z2):
        if dist.is_initialized():
            p1 = concat_all_gather(p1)
            z2 = concat_all_gather(z2)
            p2 = concat_all_gather(p2)
            z1 = concat_all_gather(z1)
        batch_size = p1.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool)
        # cross sim p1 z2
        cross_similarity_p1_z2 = self._calculate_cosine_similarity(p1, z2)
        cross_similarity_p1_z2 = cross_similarity_p1_z2 / self.temperature
        cross_similarity_p1_z2 = cross_similarity_p1_z2.exp()
        diag_elems_1 = torch.diagonal(cross_similarity_p1_z2, 0)
        row_sum_cross_p1_z2 = cross_similarity_p1_z2.sum(0)  # Taking sum across row
        row_sum_cross_p1_z2 = row_sum_cross_p1_z2 - diag_elems_1

        #cross sim p2 z1
        cross_similarity_p2_z1 = self._calculate_cosine_similarity(p2, z1)
        cross_similarity_p2_z1 = cross_similarity_p2_z1 / self.temperature
        cross_similarity_p2_z1 = cross_similarity_p2_z1.exp()
        diag_elems_2 = torch.diagonal(cross_similarity_p2_z1, 0)
        row_sum_cross_p2_z1 = cross_similarity_p2_z1.sum(0)  # Taking sum across row
        row_sum_cross_p2_z1 = row_sum_cross_p2_z1 - diag_elems_2

        row_sum_cross = row_sum_cross_p1_z2 + row_sum_cross_p2_z1 
        # Isolating the diagonal elements because we expect the positive
        # elements to be in the diagonals


        if self.use_row_sum_a:

            a_similarity_p1 = self._calculate_cosine_similarity(p1, p1)
            a_similarity_p2 = self._calculate_cosine_similarity(p2, p2)

        # print(type(a_similarity))
        # print(a_similarity)
            a_similarity_p1[mask] = 0.
            a_similarity_p1 = a_similarity_p1 / self.temperature
            a_similarity_p1 = a_similarity_p1.exp()
            row_sum_a_p1 = a_similarity_p1.sum(0)

            a_similarity_p2[mask] = 0.
            a_similarity_p2 = a_similarity_p2 / self.temperature
            a_similarity_p2 = a_similarity_p2.exp()
            row_sum_a_p2 = a_similarity_p2.sum(0)
            row_sum_a = row_sum_a_p1 + row_sum_a_p2

        if self.use_row_sum_b:
            b_similarity_z1 = self._calculate_cosine_similarity(z1, z1)
            b_similarity_z1[mask] = 0.
            b_similarity_z1 = b_similarity_z1 / self.temperature
            b_similarity_z1 = b_similarity_z1.exp()
            b_similarity_z1 = b_similarity_z1.sum(0)

            b_similarity_z2 = self._calculate_cosine_similarity(z2, z2)
            b_similarity_z2[mask] = 0.
            b_similarity_z2 = b_similarity_z2 / self.temperature
            b_similarity_z2 = b_similarity_z2.exp()
            b_similarity_z2 = b_similarity_z2.sum(0)

            row_sum_b = b_similarity_z1 + b_similarity_z2




        diag_elems = diag_elems_1 + diag_elems_2
        # We are taking
        denominator_1 = row_sum_cross_p1_z2
        denominator_2 = row_sum_cross_p2_z1
        if self.use_row_sum_a:
            denominator += row_sum_a
        if self.use_row_sum_b:
            denominator += row_sum_b
        if self.use_positives_in_denominator:
            denominator_1 += diag_elems_1
            denominator_2 += diag_elems_2

        denominator_1 += 1e-8
        denominator_2 += 1e-8
        loss_1 = - torch.log(diag_elems_1 / denominator_1).mean()
        loss_2 = - torch.log(diag_elems_2 / denominator_2).mean()
        #just like simsiam loss divide 2
        loss = loss_1/2 + loss_2/2
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else: 
            ret_dict = {f'{self.name}_contrastive_loss': loss}

        return ret_dict




@LOSSES.register_module()
class Symmetric_ContrastiveLossv2(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5, name=None, # other contrastiv method temperature=0.07
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


    def _forward(self, p1, p2, z1, z2):
        if dist.is_initialized():
            p1 = concat_all_gather(p1)
            z2 = concat_all_gather(z2)
            p2 = concat_all_gather(p2)
            z1 = concat_all_gather(z1)
        batch_size = p1.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool)
        # cross sim p1 z2
        cross_similarity_p1_z2 = self._calculate_cosine_similarity(p1, z2)
        cross_similarity_p1_z2 = cross_similarity_p1_z2 / self.temperature
        cross_similarity_p1_z2 = cross_similarity_p1_z2.exp()
        diag_elems_1 = torch.diagonal(cross_similarity_p1_z2, 0)
        row_sum_cross_p1_z2 = cross_similarity_p1_z2.sum(1)  # Taking sum across row
        row_sum_cross_p1_z2 = row_sum_cross_p1_z2 - diag_elems_1

        #cross sim p2 z1
        cross_similarity_p2_z1 = self._calculate_cosine_similarity(p2, z1)
        cross_similarity_p2_z1 = cross_similarity_p2_z1 / self.temperature
        cross_similarity_p2_z1 = cross_similarity_p2_z1.exp()
        diag_elems_2 = torch.diagonal(cross_similarity_p2_z1, 0)
        row_sum_cross_p2_z1 = cross_similarity_p2_z1.sum(1)  # Taking sum across row
        row_sum_cross_p2_z1 = row_sum_cross_p2_z1 - diag_elems_2

        row_sum_cross = row_sum_cross_p1_z2 + row_sum_cross_p2_z1 
        # Isolating the diagonal elements because we expect the positive
        # elements to be in the diagonals


        if self.use_row_sum_a:

            a_similarity_p1 = self._calculate_cosine_similarity(p1, p1)
            a_similarity_p2 = self._calculate_cosine_similarity(p2, p2)

        # print(type(a_similarity))
        # print(a_similarity)
            a_similarity_p1[mask] = 0.
            a_similarity_p1 = a_similarity_p1 / self.temperature
            a_similarity_p1 = a_similarity_p1.exp()
            row_sum_a_p1 = a_similarity_p1.sum(0)

            a_similarity_p2[mask] = 0.
            a_similarity_p2 = a_similarity_p2 / self.temperature
            a_similarity_p2 = a_similarity_p2.exp()
            row_sum_a_p2 = a_similarity_p2.sum(0)
            row_sum_a = row_sum_a_p1 + row_sum_a_p2

        if self.use_row_sum_b:
            b_similarity_z1 = self._calculate_cosine_similarity(z1, z1)
            b_similarity_z1[mask] = 0.
            b_similarity_z1 = b_similarity_z1 / self.temperature
            b_similarity_z1 = b_similarity_z1.exp()
            b_similarity_z1 = b_similarity_z1.sum(0)

            b_similarity_z2 = self._calculate_cosine_similarity(z2, z2)
            b_similarity_z2[mask] = 0.
            b_similarity_z2 = b_similarity_z2 / self.temperature
            b_similarity_z2 = b_similarity_z2.exp()
            b_similarity_z2 = b_similarity_z2.sum(0)

            row_sum_b = b_similarity_z1 + b_similarity_z2




        #diag_elems = diag_elems_1 + diag_elems_2
        # We are taking
        denominator_1 = row_sum_cross_p1_z2
        denominator_2 = row_sum_cross_p2_z1
        if self.use_row_sum_a:
            denominator += row_sum_a
        if self.use_row_sum_b:
            denominator += row_sum_b
        if self.use_positives_in_denominator:
            denominator_1 += diag_elems_1
            denominator_2 += diag_elems_2

        denominator_1 += 1e-8
        denominator_2 += 1e-8
        loss_1 = - torch.log(diag_elems_1 / denominator_1).mean()
        loss_2 = - torch.log(diag_elems_2 / denominator_2).mean()
        #just like simsiam loss divide 2
        loss = loss_1+ loss_2
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else: 
            ret_dict = {f'{self.name}_contrastive_loss': loss}

        return ret_dict



@LOSSES.register_module()
class Asymmetric_ContrastiveLossv2(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5, name=None, # other contrastiv method temperature=0.07
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


    def _forward(self, p1, z2):
        if dist.is_initialized():
            p1 = concat_all_gather(p1)
            z2 = concat_all_gather(z2)

        batch_size = p1.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool)
        # cross sim p1 z2
        row_sum_cross = self._calculate_cosine_similarity(p1, z2)
        row_sum_cross = row_sum_cross / self.temperature
        row_sum_cross = row_sum_cross.exp()
        diag_elems = torch.diagonal(row_sum_cross, 0)
        row_sum_cross = row_sum_cross.sum(0)  # Taking sum across row
        row_sum_cross = row_sum_cross - diag_elems

    

      
        # Isolating the diagonal elements because we expect the positive
        # elements to be in the diagonals


        if self.use_row_sum_a:

            a_similarity_p1 = self._calculate_cosine_similarity(p1, p1)
      

        # print(type(a_similarity))
        # print(a_similarity)
            a_similarity_p1[mask] = 0.
            a_similarity_p1 = a_similarity_p1 / self.temperature
            a_similarity_p1 = a_similarity_p1.exp()
            row_sum_a = a_similarity_p1.sum(0)

       
        if self.use_row_sum_b:
            b_similarity_z2 = self._calculate_cosine_similarity(z2, z2)
            b_similarity_z2[mask] = 0.
            b_similarity_z2 = b_similarity_z2 / self.temperature
            b_similarity_z2 = b_similarity_z2.exp()
            b_similarity_z2 = b_similarity_z2.sum(0)

            row_sum_b = b_similarity_z2




      
        # We are taking
        denominator = row_sum_cross
      
        if self.use_row_sum_a:
            denominator += row_sum_a
        if self.use_row_sum_b:
            denominator += row_sum_b
        if self.use_positives_in_denominator:
            denominator += diag_elems
         

        denominator += 1e-8
  
        #just like simsiam loss divide 2
        loss = - torch.log(diag_elems / denominator).mean()
        loss = loss * 2
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else: 
            ret_dict = {f'{self.name}_contrastive_loss': loss}

        return ret_dict
