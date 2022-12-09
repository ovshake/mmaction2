# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import torch.distributed as dist

from ..builder import LOSSES
from .base import BaseWeightedLoss

def concat_all_gather(tensor, dim=0):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
		for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
	tensors_gather[dist.get_rank()] = tensor
	output = torch.cat(tensors_gather, dim=dim)
	return output



@LOSSES.register_module()
class multi_contrastive_loss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.7, #previous 0.5 moco-v2 - 0.07 
                use_row_sum_a=False,
                use_row_sum_b=False,
                use_positives_in_denominator=False,):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
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



    def _calculate_one_invariant_info_nce_loss(self, features):
        if dist.is_initialized():

            features = concat_all_gather(features, dim=1)
        batch_size = features[0].shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool)
        q_feat = features[0]
        k_0_feat = features[1]
        k_1_feat = features[2]
        k_2_feat = features[3]
        num_embedding_space = len(features)
        loss = 0.0

        similarity_matrix_q_k0 = self._calculate_cosine_similarity(q_feat, k_0_feat)
        similarity_matrix_q_k0 = similarity_matrix_q_k0 / self.temperature
        similarity_matrix_q_k0 = similarity_matrix_q_k0.exp()
        similarity_matrix_q_k1 = self._calculate_cosine_similarity(q_feat, k_1_feat)
        similarity_matrix_q_k1 = similarity_matrix_q_k1 / self.temperature
        similarity_matrix_q_k1 = similarity_matrix_q_k1.exp()
        similarity_matrix_q_k2 = self._calculate_cosine_similarity(q_feat, k_2_feat)
        similarity_matrix_q_k2 = similarity_matrix_q_k2 / self.temperature
        similarity_matrix_q_k2 = similarity_matrix_q_k2.exp()

        q_similarity = self._calculate_cosine_similarity(q_feat, q_feat)
        q_similarity[mask] = 0.0
        q_similarity = q_similarity / self.temperature
        q_similarity = q_similarity.exp()

        k0_similarity = self._calculate_cosine_similarity(k_0_feat, k_0_feat)
        k0_similarity[mask] = 0.0
        k0_similarity = k0_similarity / self.temperature
        k0_similarity = k0_similarity.exp()
        k1_similarity = self._calculate_cosine_similarity(k_1_feat, k_1_feat)
        k1_similarity[mask] = 0.0
        k1_similarity = k1_similarity / self.temperature
        k1_similarity = k1_similarity.exp()
        k2_similarity = self._calculate_cosine_similarity(k_2_feat, k_2_feat)
        k2_similarity[mask] = 0.0
        k2_similarity = k2_similarity / self.temperature
        k2_similarity = k2_similarity.exp()

        for idx in range(1, num_embedding_space):
            similarity_matrix = self._calculate_cosine_similarity(q_feat, features[idx])
            diag_elems = torch.diagonal(similarity_matrix, 0)
            row_sum = similarity_matrix.sum(0)  # Taking sum across row
            col_sum = similarity_matrix.sum(1)  # Taking sum across columns
            a_similarity = self._calculate_cosine_similarity(q_feat, q_feat)
            a_similarity[mask] = 0.
            b_similarity = self._calculate_cosine_similarity(features[idx], features[idx])
            b_similarity[mask] = 0.

            similarity_matrix = similarity_matrix / self.temperature
            similarity_matrix = similarity_matrix.exp()
            a_similarity = a_similarity / self.temperature
            a_similarity = a_similarity.exp()

            b_similarity = b_similarity / self.temperature
            b_similarity = b_similarity.exp()
            row_sum_a = a_similarity.sum(0)
            row_sum_b = b_similarity.sum(0)
            negatives_from_other_subspace = 1e-8
            denominator = 1e-8
            if self.use_row_sum_a:
                denominator += row_sum_a
            if self.use_row_sum_b:
                denominator += row_sum_b
            if self.use_positives_in_denominator:
                denominator += row_sum


            # print(type(diag_elems))
            # print(diag_elems)
            # print(type(denominator))
            #print(denominator)
            #print((torch.log(diag_elems / (denominator))))
            # print(torch.isnan(diag_elems))
            # print('---------------------')
            # print(torch.isnan(denominator))
            loss += - (
                    torch.log(diag_elems/denominator)/3
            )
        # print(torch.isnan(loss))
        ret_dict = {"Zk_contrastive_loss": loss}
        return ret_dict

    def _forward(self, features):
        loss_dict = {}

        zk_loss = self._calculate_one_invariant_info_nce_loss(features)
        # all_loss = - (1/3)*(loss_z0 + loss_z1_2)
        # loss = {"Multi_contrastive_loss":all_loss }
        # loss_dict.update(loss)

        loss_dict.update(zk_loss)
        return loss_dict