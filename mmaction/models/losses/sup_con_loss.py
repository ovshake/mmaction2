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

    def __init__(self, num_classes, temperature=.3, loss_weight=1., normalize_feature=False, **kwargs):
        super().__init__(loss_weight)
        self.num_classes = num_classes
        self.temperature = temperature
        self.normalize_feature=normalize_feature
        self.loss = SupConLoss(self.temperature, base_temperature=self.temperature)
      


    def _forward(self, q1, k1, q2, k2, label, **kwargs):
        if self.normalize_feature:
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
            ret_dict = {'symmetric_contrastive_loss': loss}
        else:
            ret_dict = {f'{self.name}_symmetric_contrastive_loss': loss}
        return ret_dict

#--------------- Asymmetric loss ------------------#
@LOSSES.register_module()
class supervisedContrastiveLoss(BaseWeightedLoss):

    def __init__(self, num_classes, temperature=.3, loss_weight=1., normalize_feature=False, **kwargs):
        super().__init__(loss_weight)
        self.num_classes = num_classes
        self.temperature = temperature
        self.normalize_feature=normalize_feature
        self.loss = SupConLoss(self.temperature, base_temperature=self.temperature)
      


    def _forward(self, features_a, features_b, label, **kwargs):
        if self.normalize_feature:
            features_a = normalize(features_a, dim=1)
            features_b = normalize(features_b, dim=1)
        # features_a  is 96,2048 or N*batch, 2048
        # features_b is 96,2048 or N*batch, 2048
        features_a = features_a.unsqueeze(1)
        # this code will make the features_a to be 96,1,2048
        features_b = features_b.unsqueeze(1)
        # this code will make the features_b to be 96,1,2048

        input_feature = torch.cat((features_a, features_b), dim=1)
        # this code will make the input_feature to be 96,2,2048

        '''feature_a is the  feature from the no stop grad pathway
        Feature_b is the feature from the stop grad pathway'''
        #need to reshape features_a and features_b to be [bsz, n_views, ...]
        # for supcon the label will be available
        # but for contrastive learning the label will not be available
        loss = self.loss(input_feature, label)

        if not self.name:
            ret_dict = {'contrastive_loss': loss}
        else:
            ret_dict = {f'{self.name}_contrastive_loss': loss}
        return ret_dict


class SupConLoss(torch.nn.Module):
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
        #changedc this part 
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))
        device = torch.device('cuda')
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
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

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

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
