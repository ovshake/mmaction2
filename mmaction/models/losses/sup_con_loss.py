import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from ..builder import LOSSES
from .base import BaseWeightedLoss
from torch.nn.functional import normalize


#--------------- Symmetric loss ------------------#

@LOSSES.register_module()
class symmetric_supervisedContrastiveLoss(BaseWeightedLoss):

    def __init__(self, temperature=.3, loss_weight=1.,type_loss='contrastive', **kwargs):
        super().__init__(loss_weight)
    
        self.type_loss=type_loss
        self.temperature = temperature
     
        self.loss = SupConLoss(self.temperature, base_temperature=self.temperature)
      


    def _forward(self, q1, k1, q2, k2, label, **kwargs):
  
        q1 = normalize(q1, dim=1)
        k1 = normalize(k1, dim=1)
        q2 = normalize(q2, dim=1)
        k2 = normalize(k2, dim=1)
        # features_a  is 96,2048 or N*batch, 2048
        # features_b is 96,2048 or N*batch, 2048
        q1 = q1.unsqueeze(1)
        k1 = k1.unsqueeze(1)
        q2 = q2.unsqueeze(1)
        k2 = k2.unsqueeze(1)
        # this code will make the features_b to be 96,1,2048
        input_1 = torch.cat((q1, k1), dim=1)
        input_2 = torch.cat((q2, k2), dim=1)

        # this code will make the input_feature to be 96,2,2048

        '''feature_a is the  feature from the no stop grad pathway
        Feature_b is the feature from the stop grad pathway'''
        #need to reshape features_a and features_b to be [bsz, n_views, ...]
        # for supcon the label will be available
        # but for contrastive learning the label will not be available
        loss_1 = self.loss(input_1, label)
        loss_2 = self.loss(input_2, label)
        loss = (loss_1 + loss_2)*0.5

        if not self.name:
            if self.type_loss=='contrastive':
                ret_dict = {'symmetric_contrastive_loss': loss}
            else:
                ret_dict = {'symmetric_supCon_loss': loss}
        else:
            if self.type_loss=='contrastive':
                ret_dict = {f'{self.name}_symmetric_contrastive_loss': loss}
            else:
                ret_dict = {f'{self.name}_symmetric_supCon_loss': loss}
        return ret_dict

#--------------- Asymmetric loss ------------------#
@LOSSES.register_module()
class supervisedContrastiveLoss(BaseWeightedLoss):

    def __init__(self, temperature=.3, loss_weight=1.,name='color',type_loss=None, **kwargs):
        super().__init__(loss_weight)
       
        self.temperature = temperature
        self.type_loss=type_loss
        self.loss = SupConLoss(self.temperature)
        self.name=name
      


    def _forward(self, features_a, features_b, label='contrastive', **kwargs):

        features_a = normalize(features_a, dim=1)
        features_b = normalize(features_b, dim=1)

        # features_a  is 96,2048 or N*batch, 2048
        # features_b is 96,2048 or N*batch, 2048
        # B = features_a.shape[0] // 4
        # features_a, features_b = features_a[:2*B:2], features_b[1:2*B:2]
    
      
        features_a = features_a.unsqueeze(1)
        # print('features_a shape - ', features_a.shape)
        # # this code will make the features_a to be 96,1,2048
        features_b = features_b.unsqueeze(1)
        # # this code will make the features_b to be 96,1,2048

        input_feature = torch.cat((features_a, features_b), dim=1)
        # print('input_feature shape - ', input_feature.shape)
        # print('input_feature - ', input_feature.shape)
        # this code will make the input_feature to be 96,2,2048

        '''feature_a is the  feature from the no stop grad pathway
        Feature_b is the feature from the stop grad pathway'''
        #need to reshape features_a and features_b to be [bsz, n_views, ...]
        # for supcon the label will be available
        # but for contrastive learning the label will not be available
        # print(input_feature)
        # print('label - ', label)
        # if torch.isnan(input_feature).any():
        #     print("The tensor contains NaN values.")
        loss = self.loss(input_feature, label)
        # print('loss - ',loss)
        if not self.name:
            if self.type_loss=='contrastive':
                ret_dict = {'contrastive_loss': loss}
            else:
                ret_dict = {'supCon_loss': loss}
        else:
            if self.type_loss=='contrastive':
                ret_dict = {f'{self.name}_contrastive_loss': loss}
            else:
                ret_dict = {f'{self.name}_supCon_loss': loss}
        return ret_dict


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print('contrast_feature', contrast_feature.shape)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print('anchor_dot_contrast', anchor_dot_contrast)
        # print('anchor_dot_contrast shape', anchor_dot_contrast.shape)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print('logits_max', logits_max.shape)
        # print('logits shape', logits.shape)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print("exp_logits: ", exp_logits)  
        # print("exp_logits.sum(1, keepdim=True): ", exp_logits.sum(1, keepdim=True))

        # print('mask * log_prob', mask * log_prob)
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # (mask * log_prob).sum(1) = NAN
        #mask.sum(1) no problem
        #print('mask.sum(1)', mask.sum(1))
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        if torch.isnan(loss).any():
            assert print("The tensor contains NaN values.")

        return loss