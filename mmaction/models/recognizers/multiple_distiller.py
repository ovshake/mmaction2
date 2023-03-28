# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .base import AvgConsensus
from .. import builder
from einops import rearrange
from .recognizer2d import Recognizer2D
from torch.nn.functional import normalize
import torch.distributed as dist

from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, get_dist_info)
import numpy as np




@RECOGNIZERS.register_module()
class Multiple_Distiller_Recognizer(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 type_loss='logit',
                 contrastive_head=None,
                 emb_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 domain=None,
                 distil_type='frame'):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.type_loss=type_loss
        self.distil_type = distil_type

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
#---------------------------------------------------------
        if self.distil_type=='video':
            consensus=dict(type='AvgConsensus', dim=1)
            consensus_ = consensus.copy()

            consensus_type = consensus_.pop('type')
            if consensus_type == 'AvgConsensus':
                self.consensus = AvgConsensus(**consensus_)

#------------------------------------------------------------
#load the teacher model
        if speed_network != None:
            self.speed_network = builder.build_model(speed_network)
            speed_network_state_dict = torch.load(speed_ckpt_path[domain])
            print(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        else:
            self.speed_network=speed_network

        if color_network != None:
            #color_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1/tsm-k400-color-contrastive_xd_sgd_color_temp_50/train_{domain}_test_{domain}/latest.pth"
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(color_ckpt_path[domain])
            print(color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        else:
            self.color_network=color_network

        if vcop_network != None:
            #vcop_ckpt_path='/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop/tsm-k400-vcop/train_{domain}_test_{domain}/latest.pth'
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            print(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
        else:
            self.vcop_network = vcop_network
        # build loss function
        self.emb_loss = builder.build_loss(emb_loss)# different type of loss can be used. 

        self.backbone_from = 'mmaction2'
        if backbone['type'].startswith('mmcls.'):
            try:
                import mmcls.models.builder as mmcls_builder
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            backbone['type'] = backbone['type'][6:]
            self.backbone = mmcls_builder.build_backbone(backbone)
            self.backbone_from = 'mmcls'
        elif backbone['type'].startswith('torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[12:]
            self.backbone = torchvision.models.__dict__[backbone_type](
                **backbone)
            # disable the classifier
            self.backbone.classifier = nn.Identity()
            self.backbone.fc = nn.Identity()
            self.backbone_from = 'torchvision'
        elif backbone['type'].startswith('timm.'):
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm to use this '
                                  'backbone.')
            backbone_type = backbone.pop('type')[5:]
            # disable the classifier
            backbone['num_classes'] = 0
            self.backbone = timm.create_model(backbone_type, **backbone)
            self.backbone_from = 'timm'
        else:
            self.backbone = builder.build_backbone(backbone)

        if neck is not None:
            self.neck = builder.build_neck(neck)

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

        self.init_weights()
        self.fp16_enabled = True

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
        #print('imgs',imgs.shape)
        return self.forward_test(imgs, **kwargs)


    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
     
     
        t_embs = []
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
                    t_embs.append(t_speed_emb)
                if self.color_network:
                    

                    t_color_emb = self.color_network.extract_feat(imgs)
             
                    t_color_emb = self.color_network.cls_head(t_color_emb.float(),num_segs ).detach()
                    assume_logit = t_color_emb[0]
                    # logit = self.color_network(imgs, num_segs)

                    print('assume_logit -----',assume_logit )
                    # print('logit -----',logit )


                    t_embs.append(t_color_emb)
                if self.vcop_network:
                    
                    t_vcop_emb = self.vcop_network.extract_feat(imgs)
                    # t_vcop_emb = nn.AdaptiveAvgPool2d(1)(t_vcop_emb)
                    # t_vcop_emb =t_vcop_emb.squeeze()
                    t_vcop_emb = self.vcop_network.cls_head(t_vcop_emb.float(),num_segs ).detach()

                    t_embs.append(t_vcop_emb)

            else:
                if self.speed_network:
                    t_speed_emb = self.speed_network.extract_feat(imgs)
                    t_speed_emb = nn.AdaptiveAvgPool2d(1)(t_speed_emb)
                    t_speed_emb =t_speed_emb.squeeze().detach()

                    t_embs.append(t_speed_emb)
                if self.color_network:
                    t_color_emb = self.color_network.extract_feat(imgs)
                    t_color_emb = nn.AdaptiveAvgPool2d(1)(t_color_emb)
                    t_color_emb =t_color_emb.squeeze().detach()

                    t_embs.append(t_color_emb)
                if self.vcop_network:
                    t_vcop_emb = self.vcop_network.extract_feat(imgs)
                    t_vcop_emb = nn.AdaptiveAvgPool2d(1)(t_vcop_emb)
                    t_vcop_emb =t_vcop_emb.squeeze().detach()

                    t_embs.append(t_vcop_emb) # the output feature from the vcop trained tsm not head .... 
            
            # print("t_embs[0] --- teacher embedding size", t_embs[0].shape)

        # batches = imgs.shape[0]
        # imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # imgs reshape to (batches*frames, c, h, w)???
        # num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs) # is this tsm backbone that aare are extracting the features from ???
        #print('x = self.extract_feat(imgs) ---- this goes into the student ', type(x))

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze()
        
        cls_score = self.cls_head(x.float(), num_segs)
        if self.distil_type=='video':
            x = self.consensus(x)
        s_embs = []
        if self.type_loss =='logit':
            if self.speed_network:
                s_embs.append(cls_score)
            if self.color_network:
                s_embs.append(cls_score)
            if self.vcop_network:
                s_embs.append(cls_score)
        else:
            if self.speed_network:
                s_embs.append(x)
            if self.color_network:
                s_embs.append(x)
            if self.vcop_network:
                s_embs.append(x)


        c1 = s_embs[0].detach().cpu().numpy()
        c2 = t_embs[0].detach().cpu().numpy()
        if np.all(c1==c2):
            print('student and teacher embeddings are same')
        # if t_embs==s_embs:
        #     print('same')
        # print(torch.equal(t_embs, s_embs))
        # print('t_embs shape', t_embs[0].shape)
        # # print('len of t_embs', len(t_embs))
        # print('s_embs shape', s_embs[0].shape)
        # print(s_embs[0])
       
        loss_emb = self.emb_loss(t_embs[0], s_embs[0])
        # for x in range(len(t_embs)):

        
        #loss_emb = self.emb_loss(t_embs, s_embs)
 
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        losses.update(loss_emb)

        losses.update(loss_cls)
        
        return losses













