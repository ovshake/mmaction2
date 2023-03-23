import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
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
class Supervised_contrastive_loss(BaseWeightedLoss):
    def __init__(self, loss_weight=1.0, temperature=0.2, name=None, # other contrastiv method temperature=0.07
                use_row_sum_a=False,
                use_row_sum_b=False,
                use_positives_in_denominator=False, 
         
                ):
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
    # def similarity(self, a,b):
    #     a_norm = F.normalize(a, dim=1,eps=1e-8)
    #     b_norm = F.normalize(b, dim=1,eps=1e-8)
    #     sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    #     return sim_mt

    def _forward(self, features_a, features_b, gt_labels=None):
        #------------------------------------
        device = (torch.device('cuda'))
        #------------------------------------
        if dist.is_initialized():
            features_a = concat_all_gather(features_a)
            features_b = concat_all_gather(features_b)
            if gt_labels != None:
                gt_labels = concat_all_gather(gt_labels)
                # mask_label = torch.eq(gt_labels, gt_labels.T).float().to(device)
        
        batch_size = features_a.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool)
        # print('features_b shape', features_b.shape)
        # if gt_labels != None:
        #     gt_labels = concat_all_gather(gt_labels)
        #     mask_label = torch.eq(gt_labels, gt_labels.T).float().to(device)
        #     print('mask_label - ', mask_label)


        cross_similarity = self._calculate_cosine_similarity(features_a, features_b)
        # print('cross_similarity size ', cross_similarity.shape)
        if self.use_row_sum_a:
            a_similarity = self._calculate_cosine_similarity(features_a, features_a)
            a_similarity[mask] = 0.
            a_similarity = a_similarity / self.temperature
            a_similarity = a_similarity.exp()
            row_sum_a = a_similarity.sum(0)
        if self.use_row_sum_b:
            b_similarity = self._calculate_cosine_similarity(features_b, features_b)
            b_similarity[mask] = 0.
            b_similarity = b_similarity / self.temperature
            b_similarity = b_similarity.exp()
            row_sum_b = b_similarity.sum(0)

        cross_similarity = cross_similarity / self.temperature
        cross_similarity = cross_similarity.exp()


        label_list = gt_labels.tilist()
        all_positive = []
        for i in range(cross_similarity.shape[0]):
            row = cross_similarity[i]

            for j in range(len(label_list)):
                row_positive = 0
                positive_class = label_list[j]
                positive_class_idx = np.where(np.array(label_list)==positive_class)[0]
                for x in positive_class_idx:
                    row_positive += row[x]
                all_positive.append(row_positive)

        sup_all_positive_diag_elems = torch.tensor(all_positive)








        # Isolating the diagonal elements because we expect the positive
        # elements to be in the diagonals
        # diag_elems = torch.diagonal(cross_similarity, 0)
        # print(diag_elems)
        # cross_similarity[mask]=0.0
        # print(cross_similarity)
        #cross_similarity = cross_similarity[mask]=0
        row_sum_cross = cross_similarity.sum(1)  # Taking sum across row
        # row_sum_cross_col = cross_similarity.sum(0)
      

        # row_sum_cross = row_sum_cross - diag_elems# added 
        # row_sum_cross_col = row_sum_cross_col - diag_elems# added
        # print('row',row_sum_cross)
        # print('column',row_sum_cross_col)

        # We are taking
        denominator = row_sum_cross
        # if self.use_row_sum_a:
        #     denominator += row_sum_a
        # if self.use_row_sum_b:
        #     denominator += row_sum_b
        # if self.use_positives_in_denominator:
        #     denominator += diag_elems

        denominator += 1e-8

        loss = - torch.log(sup_all_positive_diag_elems / denominator).mean()
        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else:
            ret_dict = {f'{self.name}_contrastive_loss': loss}
        return ret_dict
