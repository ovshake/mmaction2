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
class LateFusionRecognizer_all(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 vcop_head=None,
                 domain=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.speed_network=speed_network
        self.color_network=color_network
        self.vcop_network=vcop_network
        print('*******************************')
        print(domain, '<<<<<=====domain')
        print('*******************************')
        
        

        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        if speed_network:

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
            self.speed_network = builder.build_model(speed_network)
            speed_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        if color_network:
            
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        if vcop_network:
            
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
            # vcop_head = dict(type='VCOPHead',num_clips=3, feature_size=2048*7*7 )
            # self.vcop_emb_head = builder.build_head(vcop_head)



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
        imgs_slow, imgs_fast,img_mid = data_batch[0]['imgs'], data_batch[1]['imgs'], data_batch[2]['imgs']
        label = data_batch[0]['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self((imgs_slow, imgs_fast,img_mid), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

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

            imgs_pathway_A, imgs_pathway_B, imgs_pathway_C = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B,imgs_pathway_C, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, speed_imgs,vcop_imgs, gt_labels, **kwargs):
        assert self.with_cls_head

        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        speed_imgs = speed_imgs.reshape((-1, ) + speed_imgs.shape[2:])
        vcop_imgs = vcop_imgs.reshape((-1,) + vcop_imgs.shape[2:])

        num_segs = color_imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(color_imgs)
        speed_features = self.color_network.extract_feat(speed_imgs)
        vcop_features = self.vcop_network.extract_feat(vcop_imgs)
   
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
      
        #normalize features before concat
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)



        losses = dict()
        fused_features = torch.cat((color_features, speed_features,vcop_features), dim=1)
        #fused_features = F.normalize(fused_features, dim=1, eps=1e-8) # normalizing the concatenated feature 
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(imgs)
        speed_features = self.color_network.extract_feat(imgs)
        vcop_features = self.vcop_network.extract_feat(imgs)

        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)

        fused_features = torch.cat((color_features, speed_features,vcop_features), dim=1)
        #fused_features = F.normalize(fused_features, dim=1, eps=1e-8) # normalizing the concatenated feature 


        # fused_features = fused_features.norm(dim=None)
    
        # fused_features = fused_features.norm(dim=1)[:, None]

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score
#-----------------------------------------------------------------------

@RECOGNIZERS.register_module()
class LateFusionRecognizer_norm_after(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 vcop_head=None,
                 domain=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.speed_network=speed_network
        self.color_network=color_network
        self.vcop_network=vcop_network

        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        if speed_network:

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
            
            self.speed_network = builder.build_model(speed_network)
            speed_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        if color_network:
            
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        if vcop_network:
            
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
            # vcop_head = dict(type='VCOPHead',num_clips=3, feature_size=2048*7*7 )
            # self.vcop_emb_head = builder.build_head(vcop_head)



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
        imgs_slow, imgs_fast,img_mid = data_batch[0]['imgs'], data_batch[1]['imgs'], data_batch[2]['imgs']
        label = data_batch[0]['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self((imgs_slow, imgs_fast,img_mid), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

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

            imgs_pathway_A, imgs_pathway_B, imgs_pathway_C = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B,imgs_pathway_C, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, speed_imgs,vcop_imgs, gt_labels, **kwargs):
        assert self.with_cls_head

        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        speed_imgs = speed_imgs.reshape((-1, ) + speed_imgs.shape[2:])
        vcop_imgs = vcop_imgs.reshape((-1,) + vcop_imgs.shape[2:])

        num_segs = color_imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(color_imgs)
        speed_features = self.color_network.extract_feat(speed_imgs)
        vcop_features = self.vcop_network.extract_feat(vcop_imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)
        # print('vcop_features', vcop_features.shape)
        # color_features = color_features.norm(dim=None)
        # speed_features = speed_features.norm(dim=None)
        # vcop_features = vcop_features.norm(dim=None)
        # print(vcop_features.shape)
        # aaa = vcop_features.norm(dim=1)[:, None]
        # print('-------------',aaa.shape)
        #print(vcop_features.norm(dim=1)[:, None].shape)
        # print('color_features size', color_features.shape)
        # print('speed_features size', speed_features.shape)
        # print('vcop_features size', vcop_features.shape)

        losses = dict()
        fused_features = torch.cat((color_features, speed_features,vcop_features), dim=1)
        # print(fused_features[0])
        # print('fused_features before normalize', fused_features.shape)
        #fused_features = F.normalize(fused_features, dim=1)
        # print(fused_features[0])
        # print('fused_features after normalize', fused_features.shape)
        # fused_features = fused_features.norm(dim=None)
        # fused_features = fused_features.norm(dim=None)
       
        #print('--------',fused_features.shape)
        #fused_features = fused_features.norm(dim=None)[:, None]
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(imgs)
        speed_features = self.color_network.extract_feat(imgs)
        vcop_features = self.vcop_network.extract_feat(imgs)

        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)


        fused_features = torch.cat((color_features, speed_features,vcop_features), dim=1)
        # fused_features = F.normalize(fused_features, dim=1)

        # fused_features = fused_features.norm(dim=None)
    
        # fused_features = fused_features.norm(dim=1)[:, None]

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score
#-----------------------------------------------------------------------

@RECOGNIZERS.register_module()
class LateFusionRecognizer_norm_before(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 vcop_head=None,
                 domain=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.speed_network=speed_network
        self.color_network=color_network
        self.vcop_network=vcop_network
        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        if speed_network:

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
            #speed_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1/tsm-k400-speed-contrastive_xd_sgd_speed_temp_5/train_{domain}_test_{domain}/latest.pth"
            self.speed_network = builder.build_model(speed_network)
            speed_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        if color_network:
            #color_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1/tsm-k400-color-contrastive_xd_sgd_color_temp_50/train_{domain}_test_{domain}/latest.pth"
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        if vcop_network:
            #vcop_ckpt_path='/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop/tsm-k400-vcop/train_{domain}_test_{domain}/latest.pth'
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
            # vcop_head = dict(type='VCOPHead',num_clips=3, feature_size=2048*7*7 )
            # self.vcop_emb_head = builder.build_head(vcop_head)



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
        imgs_slow, imgs_fast,img_mid = data_batch[0]['imgs'], data_batch[1]['imgs'], data_batch[2]['imgs']
        label = data_batch[0]['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self((imgs_slow, imgs_fast,img_mid), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

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

            imgs_pathway_A, imgs_pathway_B, imgs_pathway_C = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B,imgs_pathway_C, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, speed_imgs,vcop_imgs, gt_labels, **kwargs):
        assert self.with_cls_head

        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        speed_imgs = speed_imgs.reshape((-1, ) + speed_imgs.shape[2:])
        vcop_imgs = vcop_imgs.reshape((-1,) + vcop_imgs.shape[2:])

        num_segs = color_imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(color_imgs)
        speed_features = self.color_network.extract_feat(speed_imgs)
        vcop_features = self.vcop_network.extract_feat(vcop_imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=None)
        speed_features = F.normalize(speed_features, dim=None)
        vcop_features = F.normalize(vcop_features, dim=None)
        # print('vcop_features', vcop_features.shape)
        # color_features = color_features.norm(dim=None)
        # speed_features = speed_features.norm(dim=None)
        # vcop_features = vcop_features.norm(dim=None)
        # print(vcop_features.shape)
        # aaa = vcop_features.norm(dim=1)[:, None]
        # print('-------------',aaa.shape)
        #print(vcop_features.norm(dim=1)[:, None].shape)
        # print('color_features size', color_features.shape)
        # print('speed_features size', speed_features.shape)
        # print('vcop_features size', vcop_features.shape)

        losses = dict()
        fused_features = torch.cat((color_features, speed_features,vcop_features), dim=1)
        # print(fused_features[0])
        # print('fused_features before normalize', fused_features.shape)
        #fused_features = F.normalize(fused_features, dim=1)
        # print(fused_features[0])
        # print('fused_features after normalize', fused_features.shape)
        # fused_features = fused_features.norm(dim=None)
        # fused_features = fused_features.norm(dim=None)
       
        #print('--------',fused_features.shape)
        #fused_features = fused_features.norm(dim=None)[:, None]
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(imgs)
        speed_features = self.color_network.extract_feat(imgs)
        vcop_features = self.vcop_network.extract_feat(imgs)

        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=None)
        speed_features = F.normalize(speed_features, dim=None)
        vcop_features = F.normalize(vcop_features, dim=None)

        fused_features = torch.cat((color_features, speed_features,vcop_features), dim=1)
        # fused_features = F.normalize(fused_features, dim=1)

        # fused_features = fused_features.norm(dim=None)
    
        # fused_features = fused_features.norm(dim=1)[:, None]

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score


#-----------------------------------------------------------------------

@RECOGNIZERS.register_module()
class LateFusionRecognizer(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 speed_network,
                 color_network,
                 domain,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        # vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
        #                     'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
        #                     'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        self.speed_network = builder.build_model(speed_network)
        self.color_network = builder.build_model(color_network)
        speed_network_state_dict = torch.load(speed_ckpt_path[domain])
        color_network_state_dict = torch.load(color_ckpt_path[domain])
        self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)

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
        imgs_slow, imgs_fast = data_batch[0]['imgs'], data_batch[1]['imgs']
        label = data_batch[0]['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self((imgs_slow, imgs_fast), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

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

            imgs_pathway_A, imgs_pathway_B = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, speed_imgs, gt_labels, **kwargs):
        assert self.with_cls_head
        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        speed_imgs = speed_imgs.reshape((-1, ) + speed_imgs.shape[2:])
        num_segs = color_imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(color_imgs)
        speed_features = self.color_network.extract_feat(speed_imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
     
        losses = dict()
        fused_features = torch.cat((color_features, speed_features), dim=1)
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(imgs)
        speed_features = self.color_network.extract_feat(imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
      
        fused_features = torch.cat((color_features, speed_features), dim=1)
        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

#------------------------------------------------------------------



@RECOGNIZERS.register_module()
class LateFusionRecognizer_vcop(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 vcop_network=None,
                 color_network=None,
                 speed_network=None,
                 domain=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        self.vcop_network = builder.build_model(vcop_network)
    
        vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
      
        self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)


        if color_network:
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        else:
            self.color_network = builder.build_model(speed_network)
            color_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)



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
        imgs_slow, imgs_fast = data_batch[0]['imgs'], data_batch[1]['imgs']
        label = data_batch[0]['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self((imgs_slow, imgs_fast), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

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

            imgs_pathway_A, imgs_pathway_B = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, vcop_imgs, gt_labels, **kwargs):
        assert self.with_cls_head
        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        vcop_imgs = vcop_imgs.reshape((-1, ) + vcop_imgs.shape[2:])
        num_segs = color_imgs.shape[0] // batches
        color_features = self.color_network.extract_feat(color_imgs)
        vcop_features = self.vcop_network.extract_feat(vcop_imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)
      
        losses = dict()
        fused_features = torch.cat((color_features, vcop_features), dim=1)
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.color_network.extract_feat(imgs)
        vcop_features = self.vcop_network.extract_feat(imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)
      
        fused_features = torch.cat((color_features, vcop_features), dim=1)
        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

#---------########################


@RECOGNIZERS.register_module()
class LateFusionRecognizer_combine_all(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 vcop_head=None,
                 domain=None,
                 fusion_type='add',
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.speed_network=speed_network
        self.color_network=color_network
        self.vcop_network=vcop_network
        self.fusion_type=fusion_type

        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        if speed_network:

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
            
            self.speed_network = builder.build_model(speed_network)
            speed_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        if color_network:
            
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        if vcop_network:
  
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
            # vcop_head = dict(type='VCOPHead',num_clips=3, feature_size=2048*7*7 )
            # self.vcop_emb_head = builder.build_head(vcop_head)



        self.cls_head = builder.build_head(cls_head) if cls_head else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg


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
        imgs_slow, imgs_fast,img_mid = data_batch[0]['imgs'], data_batch[1]['imgs'], data_batch[2]['imgs']
        label = data_batch[0]['label']


        losses = self((imgs_slow, imgs_fast,img_mid), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

        return outputs

    def forward(self, imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""

        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')

            imgs_pathway_A, imgs_pathway_B, imgs_pathway_C = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B,imgs_pathway_C, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, speed_imgs,vcop_imgs, gt_labels, **kwargs):
        assert self.with_cls_head

        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        speed_imgs = speed_imgs.reshape((-1, ) + speed_imgs.shape[2:])
        vcop_imgs = vcop_imgs.reshape((-1,) + vcop_imgs.shape[2:])

        num_segs = color_imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(color_imgs)
        speed_features = self.color_network.extract_feat(speed_imgs)
        vcop_features = self.vcop_network.extract_feat(vcop_imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)
        
        # color_features = F.normalize(color_features, dim=1)
        # speed_features = F.normalize(speed_features, dim=1)
        # vcop_features = F.normalize(vcop_features, dim=1)
  
        losses = dict()
        if self.fusion_type == 'add':
            fused_features = torch.add(torch.add(color_features, speed_features), vcop_features)
        elif self.fusion_type == 'avg':
            fused_features = torch.add(torch.add(color_features, speed_features), vcop_features) / 3
        else:
            fused_features = torch.cat((color_features, vcop_features), dim=1)

      
    
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.speed_network.extract_feat(imgs)
        speed_features = self.color_network.extract_feat(imgs)
        vcop_features = self.vcop_network.extract_feat(imgs)

        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()

        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)

        if self.fusion_type == 'add':
            fused_features = torch.add(torch.add(color_features, speed_features), vcop_features)
        if self.fusion_type == 'avg':
            fused_features = torch.add(torch.add(color_features, speed_features), vcop_features) / 3
        else:
            fused_features = torch.cat((color_features, vcop_features), dim=1)
   

        # fused_features = fused_features.norm(dim=None)
    
        # fused_features = fused_features.norm(dim=1)[:, None]

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score


#------------------------------------------------------------

@RECOGNIZERS.register_module()
class LateFusionRecognizer_combine_two(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 vcop_network=None,
                 color_network=None,
                 speed_network=None,
                 domain=None,
                 fusion_type='add',
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):   #   /data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V1/tsm-k400-color_speed_pathway
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        #color_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1/tsm-k400-color-contrastive_xd_sgd_color_temp_50/train_{domain}_test_{domain}/latest.pth"
        
        self.vcop_network = builder.build_model(vcop_network)
    
        vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
      
        self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)


        if color_network:
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        else:
            self.color_network = builder.build_model(speed_network)
            color_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        self.fusion_type =fusion_type

        self.cls_head = builder.build_head(cls_head) if cls_head else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors






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
        imgs_slow, imgs_fast = data_batch[0]['imgs'], data_batch[1]['imgs']
        label = data_batch[0]['label']

   

        losses = self((imgs_slow, imgs_fast), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

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

            imgs_pathway_A, imgs_pathway_B = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, vcop_imgs, gt_labels, **kwargs):
        assert self.with_cls_head
        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        vcop_imgs = vcop_imgs.reshape((-1, ) + vcop_imgs.shape[2:])
        num_segs = color_imgs.shape[0] // batches
        color_features = self.color_network.extract_feat(color_imgs)
        vcop_features = self.vcop_network.extract_feat(vcop_imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)
    
        losses = dict()
        if self.fusion_type =='add':
            fused_features = color_features + vcop_features
        elif self.fusion_type =='avg':
            fused_features = (color_features + vcop_features)/2
        else:
            fused_features = torch.cat((color_features, vcop_features), dim=1)
   

        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.color_network.extract_feat(imgs)
        vcop_features = self.vcop_network.extract_feat(imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)
    
        if self.fusion_type =='add':
            fused_features = color_features + vcop_features
        elif self.fusion_type =='avg':
            fused_features = (color_features + vcop_features)/2
        else:
            fused_features = torch.cat((color_features, vcop_features), dim=1)

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

#------------------------------
@RECOGNIZERS.register_module()
class LateFusionRecognizer_combine_speed_color(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 vcop_network=None,
                 color_network=None,
                 speed_network=None,
                 domain=None,
                 fusion_type='add',
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        speed_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}
        
        #color_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1/tsm-k400-color-contrastive_xd_sgd_color_temp_50/train_{domain}_test_{domain}/latest.pth"
        


        if color_network:
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        if speed_network:
            self.speed_network = builder.build_model(speed_network)
            speed_network_state_dict = torch.load(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        self.fusion_type =fusion_type

        self.cls_head = builder.build_head(cls_head) if cls_head else None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # aux_info is the list of tensor names beyond 'imgs' and 'label' which
        # will be used in train_step and val_step, data_batch should contain
        # these tensors






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
        imgs_slow, imgs_fast = data_batch[0]['imgs'], data_batch[1]['imgs']
        label = data_batch[0]['label']

   

        losses = self((imgs_slow, imgs_fast), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

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

            imgs_pathway_A, imgs_pathway_B = imgs
            return self.forward_train(imgs_pathway_A, imgs_pathway_B, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, color_imgs, speed_imgs, gt_labels, **kwargs):
        assert self.with_cls_head
        batches = color_imgs.shape[0]
        color_imgs = color_imgs.reshape((-1, ) + color_imgs.shape[2:])
        speed_imgs = speed_imgs.reshape((-1, ) + speed_imgs.shape[2:])
        num_segs = color_imgs.shape[0] // batches
        color_features = self.color_network.extract_feat(color_imgs)
        speed_features = self.speed_network.extract_feat(speed_imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
    
        losses = dict()
        if self.fusion_type =='add':
            fused_features = color_features + speed_features
        elif self.fusion_type =='avg':
            fused_features = (color_features + speed_features)/2
        else:
            fused_features = torch.cat((color_features, speed_features), dim=1)
   

        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        color_features = self.color_network.extract_feat(imgs)
        speed_features = self.speed_network.extract_feat(imgs)
        color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
        speed_features = nn.AdaptiveAvgPool2d(1)(speed_features).squeeze()
        color_features = F.normalize(color_features, dim=1, eps=1e-8)
        speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
    
    
        if self.fusion_type =='add':
            fused_features = color_features + speed_features
        elif self.fusion_type =='avg':
            fused_features = (color_features + speed_features)/2
        else:
            fused_features = torch.cat((color_features, speed_features), dim=1)

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score