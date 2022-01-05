# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F

from ..builder import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register_module()
class MultipleContrastiveLoss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        # self.all_way = all_way
    
    def _calculate_cosine_similarity(self, a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt
    
    # def _calculate_all_invariant_info_nce(self, embedding_features):
    #     batch_size = embedding_features[0].shape[0] 
    #     mask = torch.eye(batch_size, dtype=torch.bool)
    #     q_feat = embedding_features[0] 
    #     num_embedding_space = len(embedding_features)
    #     positives = 1e-8
    #     negatives = 1e-8
    #     for idx in range(1, num_embedding_space):
    #         k_feat = embedding_features[idx] 
    #         similarity = self._calculate_cosine_similarity(q_feat, k_feat) 
    #         positives += similarity[mask].sum(axis=-1)
    #         negatives += similarity[~mask].sum(axis=-1) 
        
    #     loss = torch.log(positives / (positives + negatives + 1e-8)) * (1 / num_embedding_space)
    #     return loss 

    def _calculate_all_invariant_info_nce_all_way(self, embedding_features):
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
        
        loss = torch.log(positives / (positives + negatives + 1e-8)) * (1 / num_embedding_space)
        return loss 
    

    def _calculate_leave_one_out_variant_info_nce(self, embedding_features):
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
            loss += torch.log(positives / negatives) 

        loss = loss * (1 / num_embedding_space)


        # for idx in range(1, num_embedding_space - 1):
        #     k_feat = embedding_features[idx] 
        #     similarity = self._calculate_cosine_similarity(q_feat, k_feat) 
        #     negatives += similarity[mask].sum(axis=-1) 
        
        # for idx in range(1, num_embedding_space - 1):
        #     k_feat = embedding_features[idx] 
        #     similarity = self._calculate_cosine_similarity(q_feat, k_feat)
        #     positives = 1e-8
        #     positives += similarity[mask].sum(axis=-1)
        #     embedding_same_instance_negative = negatives - positives # same instance different contrastive spaces -ves 
        #     embedding_diff_instance_negative = similarity[~mask].sum(axis=-1) # different instances same contrastive space -ves 
        #     loss += torch.log(positives / (positives + embedding_same_instance_negative + 
        #                 embedding_diff_instance_negative + 1e-8)) * (1 / num_embedding_space)
        

        return loss 
    
    def _forward(self, embedding_features):
        assert len(embedding_features) > 0, 'Atleast features from one embedding space required'
        # if not self.all_way:
        #     loss = - (self._calculate_all_invariant_info_nce(embedding_features) + 
        #             self._calculate_leave_one_out_variant_info_nce(embedding_features))
        # else:
        loss = - (self._calculate_all_invariant_info_nce_all_way(embedding_features) + 
                self._calculate_leave_one_out_variant_info_nce(embedding_features))
          
        ret_dict = {'multiple_contrastive_losses': self.loss_weight * loss} 
        return ret_dict 