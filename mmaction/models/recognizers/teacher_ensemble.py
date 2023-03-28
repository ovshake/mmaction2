# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .. import builder
from einops import rearrange
import numpy
import sys
from .recognizer2d import Recognizer2D

@RECOGNIZERS.register_module()
class Teacher_ensemble(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 type_loss='logit',
                 train_cfg=None,
                 test_cfg=None,
                 domain=None,
                 distil_type='frame'):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.speed_network=speed_network
        self.color_network=color_network
        self.vcop_network=vcop_network
        self.type_loss=type_loss
        self.distil_type=distil_type

        print('*******************************')
        print(domain, '<<<<<=====domain')
        print('*******************************')
        
        
        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/VCOP_4_clip/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/VCOP_4_clip/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/VCOP_4_clip/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_speed_pathway_w_pretrained_avg_linear/train_D1_test_D1/best_top1_acc_epoch_50.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_speed_pathway_w_pretrained_avg_linear/train_D2_test_D2/best_top1_acc_epoch_80.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_speed_pathway_w_pretrained_avg_linear/train_D3_test_D3/best_top1_acc_epoch_20.pth'}

        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_color_pathway_w_pretrained_linear/train_D1_test_D1/best_top1_acc_epoch_60.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_color_pathway_w_pretrained_linear/train_D2_test_D2/best_top1_acc_epoch_80.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_color_pathway_w_pretrained_linear/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        self.idx = 0
        if speed_network:
            self.idx +=1
            self.speed_network = builder.build_model(speed_network)
            speed_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        else:
            self.speed_network = None

        if color_network:
            self.idx +=1
            
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        else:
            self.color_network = None
        
        if vcop_network:
            self.idx +=1
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
        else:
            self.vcop_network = None


        # cls_head=dict(type='TSMHead',
        #                 num_segments=8,
        #                 num_classes=8,
        #                 spatial_type=None,
        #                 consensus=dict(type='AvgConsensus', dim=1),
        #                 in_channels=2048*self.idx,
        #                 init_std=0.001,
        #                 dropout_ratio=0.0)

        self.cls_head = builder.build_head(cls_head) if cls_head else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors
        self.aux_info = []
        if train_cfg is not None and 'aux_info' in train_cfg:
            self.aux_info = train_cfg['aux_info']
        # max_testing_views should be int
        self.max_testing_views = None
        if test_cfg is not None and 'max_testing_views' in test_cfg:
            self.max_testing_views = test_cfg['max_testing_views']
            assert isinstance(self.max_testing_views, int)

        if test_cfg is not None and 'feature_extraction' in test_cfg:
            self.feature_extraction = test_cfg['feature_extraction']
        else:
            self.feature_extraction = False

        # mini-batch blending, e.g. mixup, cutmix, etc.
        self.blending = None
        if train_cfg is not None and 'blending' in train_cfg:
            from mmcv.utils import build_from_cfg
            from mmaction.datasets.builder import BLENDINGS
            self.blending = build_from_cfg(train_cfg['blending'], BLENDINGS)


        self.fp16_enabled = False

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch['imgs']

        label = data_batch['label']
        # imgs = data_batch[0]['imgs']

        # label = data_batch[0]['label']
        losses = self(imgs, label, return_loss=True)
        


        # losses = self((imgs_slow, imgs_fast,img_mid), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))
        # outputs = dict(
        #     loss=loss,
        #     log_vars=log_vars,
        #     num_samples=len(next(iter(data_batch[0].values()))))

        return outputs

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            if self.blending is not None:
                imgs, label = self.blending(imgs, label)
    
            return self.forward_train(imgs, label, **kwargs)

        return self.forward_test(imgs, **kwargs)


        
    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
     
     
        ensemble_network = []
        # print('imgs that goes in to teacher', imgs.shape)
        with torch.no_grad():
            if self.type_loss =='logit':
                if self.speed_network:
                    t_speed_emb = self.speed_network.extract_feat(imgs)
                    # t_speed_emb = nn.AdaptiveAvgPool2d(1)(t_speed_emb)
                    # t_speed_emb =t_speed_emb.squeeze()
                    #(N x num_seg, 2048)
                    t_speed_emb = self.speed_network.cls_head(t_speed_emb.float(),num_segs ).detach()
                    #(N x num_seg, class_num)
                    ensemble_network.append(t_speed_emb)
                if self.color_network:
                    
                    t_color_emb = self.color_network.extract_feat(imgs)
                    t_color_emb = self.color_network.cls_head(t_color_emb.float(),num_segs ).detach()
                    # logit = self.color_network(imgs, num_segs)

                    ensemble_network.append(t_color_emb)
                if self.vcop_network:
                    
                    t_vcop_emb = self.vcop_network.extract_feat(imgs)
                    # t_vcop_emb = nn.AdaptiveAvgPool2d(1)(t_vcop_emb)
                    # t_vcop_emb =t_vcop_emb.squeeze()
                    t_vcop_emb = self.vcop_network.cls_head(t_vcop_emb.float(),num_segs ).detach()

                    ensemble_network.append(t_vcop_emb)

            else:
                if self.speed_network:
                    t_speed_emb = self.speed_network.extract_feat(imgs)
                    t_speed_emb = nn.AdaptiveAvgPool2d(1)(t_speed_emb)
                    t_speed_emb =t_speed_emb.squeeze().detach()

                    ensemble_network.append(t_speed_emb)
                if self.color_network:
                    t_color_emb = self.color_network.extract_feat(imgs)
                    t_color_emb = nn.AdaptiveAvgPool2d(1)(t_color_emb)
                    t_color_emb =t_color_emb.squeeze().detach()

                    ensemble_network.append(t_color_emb)
                if self.vcop_network:
                    t_vcop_emb = self.vcop_network.extract_feat(imgs)
                    t_vcop_emb = nn.AdaptiveAvgPool2d(1)(t_vcop_emb)
                    t_vcop_emb =t_vcop_emb.squeeze().detach()

                    ensemble_network.append(t_vcop_emb)



        losses = dict()
        stacked_tensor = torch.stack(ensemble_network, dim=0)
        fused_features = stacked_tensor.mean(dim=0)
        if self.type_loss =='logit':
            cls_scores = fused_features
        else:
            cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        ensemble_network = []
        # print('imgs that goes in to teacher', imgs.shape)
        with torch.no_grad():
            if self.type_loss =='logit':
                if self.speed_network:
                    t_speed_emb = self.speed_network.extract_feat(imgs)
                    # t_speed_emb = nn.AdaptiveAvgPool2d(1)(t_speed_emb)
                    # t_speed_emb =t_speed_emb.squeeze()
                    #(N x num_seg, 2048)
                    t_speed_emb = self.speed_network.cls_head(t_speed_emb.float(),num_segs ).detach()
                    #(N x num_seg, class_num)
                    ensemble_network.append(t_speed_emb)
                if self.color_network:
                    
                    t_color_emb = self.color_network.extract_feat(imgs)
                    t_color_emb = self.color_network.cls_head(t_color_emb.float(),num_segs ).detach()
                    # logit = self.color_network(imgs, num_segs)

                    ensemble_network.append(t_color_emb)
                if self.vcop_network:
                    
                    t_vcop_emb = self.vcop_network.extract_feat(imgs)
                    # t_vcop_emb = nn.AdaptiveAvgPool2d(1)(t_vcop_emb)
                    # t_vcop_emb =t_vcop_emb.squeeze()
                    t_vcop_emb = self.vcop_network.cls_head(t_vcop_emb.float(),num_segs ).detach()

                    ensemble_network.append(t_vcop_emb)

            else:
                if self.speed_network:
                    t_speed_emb = self.speed_network.extract_feat(imgs)
                    t_speed_emb = nn.AdaptiveAvgPool2d(1)(t_speed_emb)
                    t_speed_emb =t_speed_emb.squeeze().detach()

                    ensemble_network.append(t_speed_emb)
                if self.color_network:
                    t_color_emb = self.color_network.extract_feat(imgs)
                    t_color_emb = nn.AdaptiveAvgPool2d(1)(t_color_emb)
                    t_color_emb =t_color_emb.squeeze().detach()

                    ensemble_network.append(t_color_emb)
                if self.vcop_network:
                    t_vcop_emb = self.vcop_network.extract_feat(imgs)
                    t_vcop_emb = nn.AdaptiveAvgPool2d(1)(t_vcop_emb)
                    t_vcop_emb =t_vcop_emb.squeeze().detach()

                    ensemble_network.append(t_vcop_emb)


        stacked_tensor = torch.stack(ensemble_network, dim=0)
        fused_features = stacked_tensor.mean(dim=0)
        if self.type_loss =='logit':
            cls_score = fused_features
        else:
            cls_score = self.cls_head(fused_features.float(), num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score
