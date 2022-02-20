# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import torch.distributed as dist

from ..builder import LOSSES
from .base import BaseWeightedLoss

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    tensors_gather[dist.get_rank()] = tensor
    output = torch.cat(tensors_gather, dim=2)
    return output

@LOSSES.register_module()
class MultipleContrastiveLoss(BaseWeightedLoss):
    """
    Multiple contrastive loss function taken from the paper
    What Should Not Be Contrastive in Contrastive Learning - 
    (https://arxiv.org/abs/2008.05659). 
    """
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
        sim_mt = sim_mt / self.temperature
        sim_mt = sim_mt.exp() 
        return sim_mt

    def _calculate_all_invariant_info_nce_all_way(self, embedding_features):
        """
        All invariant loss function. 
        """
        batch_size = embedding_features[0].shape[0] 
        mask = torch.eye(batch_size, dtype=torch.bool)
        num_embedding_space = len(embedding_features)
        positives = 1e-8
        negatives = 1e-8
        for k1 in range(num_embedding_space):
            for k2 in range(k1 + 1, num_embedding_space):
                k1_feat = embedding_features[k1]
                k2_feat = embedding_features[k2] 
                similarity = self._calculate_cosine_similarity(k1_feat, k2_feat) 
                positives += similarity[mask].sum(axis=-1)
                negatives += similarity[~mask].sum(axis=-1)
                
        loss = torch.log(positives / (positives + negatives))
        return loss 
    

    def _calculate_leave_one_out_variant_info_nce(self, embedding_features):
        """
        Invariant to only one space. The `embedding_features are 
        ordered in [K0, K1, K2, Q] so the last features hold the Q features. 

        For only-one invariant spaces there are two negatives. 
        E.g. for Two augmentations A1 and A2 there are two keys K1 and K2 and two
        constant keys Q and K0. Now <Q, K1> are positive. <K0, K2> are negative and 
        all other samples having A1 augmentations are negative. 
        """
        batch_size = embedding_features[0].shape[0] 
        mask = torch.eye(batch_size, dtype=torch.bool)
        q_feat = embedding_features[-1] 
        k_0_feat = embedding_features[0]
        num_embedding_space = len(embedding_features)
        loss = 0.

        for idx in range(1, num_embedding_space - 1):
            similarity_matrix = self._calculate_cosine_similarity(q_feat, embedding_features[idx])
            positives = 1e-8
            negatives = 1e-8
            positives += similarity_matrix[mask].sum(axis=-1) 
            for idx_negative in range(1, num_embedding_space - 1):
                if idx != idx_negative:
                    negative_similarity_matrix = self._calculate_cosine_similarity(k_0_feat, embedding_features[idx_negative])
                    negatives += negative_similarity_matrix[mask].sum(axis=-1)

            negatives += similarity_matrix[~mask].sum(axis=-1)
            loss += torch.log(positives / (negatives + positives)) 

        return loss 
    
    def _forward(self, all_embedding_features):
        assert len(all_embedding_features) > 0, 'Atleast features from one embedding space required'
        num_embedding_space = all_embedding_features.shape[0]
        all_embedding_features = concat_all_gather(all_embedding_features)
        loss = 0.
        for idx_espace in range(1, num_embedding_space):
            loss += - self._calculate_leave_one_out_variant_info_nce(all_embedding_features[idx_espace]) 
        
        loss += - self._calculate_all_invariant_info_nce_all_way(all_embedding_features[0])  
        
        loss = loss * (1 / num_embedding_space)

        ret_dict = {'multiple_contrastive_losses': self.loss_weight * loss} 
        return ret_dict 