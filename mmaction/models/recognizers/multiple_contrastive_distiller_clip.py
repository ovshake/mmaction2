# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
import torch.nn.functional as F
from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .. import builder
from einops import rearrange
from .recognizer2d import Recognizer2D
from torch.nn.functional import normalize
import torch.distributed as dist
import clip
from PIL import Image
import torchvision.transforms as T
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, OptimizerHook,
                         build_optimizer, get_dist_info)
import numpy as np


@RECOGNIZERS.register_module()
class MultipleContrastiveDistillerRecognizer_clip_ucf_hmdb(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 type_loss='stack',
                 contrastive_head=None,
                 emb_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 domain=None,
                 clip_method=None,
                 emb_stage='proj_head'):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.clip_method = clip_method
        self.emb_stage = emb_stage
        self.type_loss=type_loss
        print('*******************************')
        print(domain, '<<<<<=====domain')
        print('*******************************')

        vcop_ckpt_path = {'ucf101': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_UCF_VCOP_clip4_new/ucf/best_top1_acc_epoch_5.pth',
                            'hmdb51': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_HMDB_VCOP_clip4_new/hmdb/best_top1_acc_epoch_15.pth'}

        speed_ckpt_path = {'ucf101':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_UCF_two_pathway_CPS_C/ucf/best_top1_acc_epoch_70.pth',
                            'hmdb51':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_HMDB_two_pathway_CPS_C/hmdb/best_top1_acc_epoch_15.pth'}

        color_ckpt_path = {'ucf101':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_UCF_two_pathway_C_S/ucf/best_top1_acc_epoch_10.pth',
                            'hmdb51':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_HMDB_two_pathway_C_S/hmdb/best_top1_acc_epoch_15.pth'}

       
       
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("RN50", device='cpu')
        self.clip_model.visual.attnpool.c_proj = torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.clip_model.to(device)
        self.clip_model.half()

        if speed_network != None:

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
           # speed_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1/tsm-k400-speed-contrastive_xd_sgd_speed_temp_5/train_{domain}_test_{domain}/latest.pth"
            self.speed_network = builder.build_model(speed_network)
            CPS_2_C_network_state_dict = torch.load(speed_ckpt_path[domain])
            print(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(CPS_2_C_network_state_dict["state_dict"], strict=False)
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

        self.emb_loss = builder.build_loss(emb_loss)

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
    
        
        #--------------------------------------------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("imgs.shape[0]", imgs.shape[0])
        print("imgs.shape[1]", imgs.shape[1])
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                for j in range(imgs.shape[1]):

                    if self.clip_method=='all':
                        single_frame = imgs[i][j]
                        if i == 0 and j == 0:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # clip_features = clip_features.detach()
                        else:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            #second_clip_features = second_clip_features.detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                            # clip_features = clip_features.detach()

                    elif self.clip_method=='center': # inflated center frame feature 
                        single_frame = imgs[i][j]
                        if i == 0 and j == 4:# in 8 frame setting, 4th frame is the center frame
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # clip_features = clip_features.detach()
                            for a in range(imgs.shape[1]-1):
                                clip_image = single_frame.unsqueeze(0).to(device)
                                second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                #second_clip_features = second_clip_features.detach()
                                clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                # clip_features = clip_features.detach()
                        elif i !=0 and j == 4:
                            for a in range(imgs.shape[1]):
                                clip_image = single_frame.unsqueeze(0).to(device)
                                second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                # second_clip_features = second_clip_features.detach()
                                clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                # clip_features = clip_features.detach()
                    else:
                        single_frame = imgs[i][j]
                        if i == 0 and j == 0:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # print('clip_features.shape 11111', clip_features.shape)
                        elif i==0 and j!=0 and j!=7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                #second_clip_features = second_clip_features.detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                        elif i==0 and j==7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                #second_clip_features = second_clip_features.detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                            # print('clip_features.shape ^88888^^^^^^ ', clip_features.shape)
                            clip_features = torch.mean(clip_features, 0,keepdim=True)
                            # print('clip_features.shape ^^^^^^^ ', clip_features.shape)
                            # clip_features = clip_features.detach()
                        else:
                            for a in range(1, imgs.shape[0]):
                                if i==a and j==0:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                elif i==a and j!=0 and j!=7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features = torch.cat((second_clip_features, third_clip_features), 0).detach()
                                elif i==a and j==7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features = torch.cat((second_clip_features, third_clip_features), 0).detach()
                                    # print('before mean second_clip_features', second_clip_features.shape)
                                    second_clip_features = torch.mean(second_clip_features, 0).unsqueeze(0)
                                    # print('after mean second_clip_features', second_clip_features.shape)
                                    
                                    # print('second_clip_features.shape ', second_clip_features.shape)
                                    clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                    # print('clip_features.shape ', clip_features.shape)

        print('clip_features - ', clip_features.shape)
       #------------------------------------------------------------------------ 
        t_clip_emb = clip_features.detach()
        t_embs = []
        t_embs.append(t_clip_emb)
        self.teacher_student = []
        # print('imgs that goes in to teacher', imgs.shape)
        # img size is torch.Size([12, 8, 3, 224, 224])
        with torch.no_grad():
            if self.speed_network:
                self.teacher_student.append('speed')
                t_speed_emb = self.speed_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                # print('teacher speed')
                # print(t_speed_emb.shape)
                t_embs.append(t_speed_emb)
            if self.color_network:
                self.teacher_student.append('color')
                t_color_emb = self.color_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                # print('teacher color')
                # print(t_color_emb.shape)
                t_embs.append(t_color_emb)
            if self.vcop_network:
                self.teacher_student.append('vcop')
                t_vcop_emb = self.vcop_network.forward_teacher(imgs, emb_stage='backbone').detach()
                # print('vcop teacher')
                # print(t_vcop_emb.shape)
                t_embs.append(t_vcop_emb) # the output feature from the vcop trained tsm not head .... 
        
        # print("t_embs[0] --- teacher embedding size", t_embs[0].shape)

        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # imgs reshape to (batches*frames, c, h, w)???
        num_segs = imgs.shape[0] // batches

        losses = dict()
        loss_emb= dict() # loss for the student

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
        #print('student size', x.shape)
        # print('student  is above with nn.AdaptiveAvgPool2d(1)(x), x.squeeze()')
        # print('x = x.squeeze() -  this goes in to the contrastive head',x.shape)
        s_embs = []
        if clip_features.shape==x.shape:
            s_embs.append(x) # append since we already have clip feature in the t_embs
        else:# x 96, 2048 
            print('x.shape', x.shape)
            for i in range(1, x.shape[0]+1):
                if i !=1 and i==8:
                    avg_x = torch.mean(x[i-8:i],0,keepdim=True)
                    print('avg_x.shape', avg_x.shape)
                elif i!=1 and i!=8 and i%8==0:
                    avg_x_second =  torch.mean(x[i-8:i], 0, keepdim=True)
                    print('avg_x_second.shape', avg_x_second.shape)
                    avg_x = torch.cat((avg_x, avg_x_second), 0)
                    print('avg_x.shape1111', avg_x.shape)
            s_embs.append(avg_x)
            if avg_x.shape!= clip_features.shape:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error - -- - - - - - - -')
        if self.speed_network:
            # s_speed_emb = self.speed_contrastive_emb_head(x.float())
            # s_embs.append(s_speed_emb)
            # print(s_speed_emb.shape)
            s_speed_emb=x
            s_embs.append(s_speed_emb)
        if self.color_network:
            s_color_emb = x #self.color_contrastive_emb_head(x.float())
            #print(s_color_emb.shape)
            s_embs.append(s_color_emb)
        if self.vcop_network:
            s_vcop_emb =x
            s_embs.append(s_vcop_emb)

        # print('s_embs[0] ---  student embedding shape ', s_embs[0].shape)

        c1 = s_embs[0].detach().cpu().numpy()
        c2 = t_embs[0].detach().cpu().numpy()
        if np.all(c1==c2):
            print('student and teacher embeddings are same')
        # if t_embs==s_embs:
        #     print('same')
        # print(torch.equal(t_embs, s_embs))
        if self.type_loss == 'stack':
            t_embs = torch.vstack(t_embs)
            s_embs = torch.vstack(s_embs)
            loss_emb = self.emb_loss(t_embs, s_embs)
        # else:
        #     if 'vcop' in self.teacher_student:
        #         vcop_loss_emb = self.emb_loss(t_vcop_emb, s_vcop_emb)
        #         loss_emb['loss_embedding']=vcop_loss_emb['loss_embedding']
        #     if 'color' in self.teacher_student:
        #         color_loss_emb = self.emb_loss(t_color_emb, s_color_emb)
        #         loss_emb['loss_embedding']=color_loss_emb['loss_embedding'] + loss_emb['loss_embedding']
        #     if 'speed' in self.teacher_student:
        #         speed_loss_emb = self.emb_loss(t_speed_emb, s_speed_emb)
        #         loss_emb['loss_embedding']=speed_loss_emb['loss_embedding'] + loss_emb['loss_embedding']

        #----------------------
            # loss_emb={}
            # color_loss_emb = self.emb_loss(t_color_emb, s_color_emb)
            # speed_loss_emb = self.emb_loss(t_speed_emb, s_speed_emb)
            # vcop_loss_emb = self.emb_loss(t_vcop_emb, s_vcop_emb)
            # loss_emb['loss_embedding']=color_loss_emb['loss_embedding']+speed_loss_emb['loss_embedding']+vcop_loss_emb['loss_embedding']
        #--------------------
        # print('t_embs --- teacher embedding shape ', t_embs.shape)
        # print('s_embs --- student embedding shape ', s_embs.shape)

        #loss_emb = self.emb_loss(t_embs, s_embs)
        cls_score = self.cls_head(x.float(), num_segs)
        # cls_score = self.cls_head(x.to(torch.float32), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        losses.update(loss_emb)

        #losses.update(loss_cls)
        
        return losses


@RECOGNIZERS.register_module()
class MultipleContrastiveDistillerRecognizer_w_clip(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 type_loss='stack',
                 contrastive_head=None,
                 emb_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 domain=None,
                 clip_method=None,
                #  dimension_reduction=None,
                 emb_stage='proj_head'):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.clip_method = clip_method
        self.emb_stage = emb_stage
        self.type_loss=type_loss
        print(self.type_loss)
        # self.dimension_reduction=dimension_reduction
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

        normal_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D1_test_D1/best_top1_acc_epoch_85.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D3_test_D3/best_top1_acc_epoch_75.pth'}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("RN50", device='cpu')
        self.clip_model.visual.attnpool.c_proj = torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.clip_model.half()
        self.clip_model.to(device)
        

        if speed_network != None:

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
           # speed_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1/tsm-k400-speed-contrastive_xd_sgd_speed_temp_5/train_{domain}_test_{domain}/latest.pth"
            self.speed_network = builder.build_model(speed_network)
            CPS_2_C_network_state_dict = torch.load(speed_ckpt_path[domain])
            print(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(CPS_2_C_network_state_dict["state_dict"], strict=False)
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

        self.emb_loss = builder.build_loss(emb_loss)

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
    
        
        #--------------------------------------------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("imgs.shape[0]", imgs.shape[0])
        print("imgs.shape[1]", imgs.shape[1])
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                for j in range(imgs.shape[1]):

                    if self.clip_method=='all':
                        single_frame = imgs[i][j]
                        if i == 0 and j == 0:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # clip_features = clip_features.detach()
                        else:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            #second_clip_features = second_clip_features.detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                            # clip_features = clip_features.detach()

                    elif self.clip_method=='center': # inflated center frame feature 
                        single_frame = imgs[i][j]
                        if i == 0 and j == 4:# in 8 frame setting, 4th frame is the center frame
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # clip_features = clip_features.detach()
                            for a in range(imgs.shape[1]-1):
                                clip_image = single_frame.unsqueeze(0).to(device)
                                second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                #second_clip_features = second_clip_features.detach()
                                clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                # clip_features = clip_features.detach()
                        elif i !=0 and j == 4:
                            for a in range(imgs.shape[1]):
                                clip_image = single_frame.unsqueeze(0).to(device)
                                second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                # second_clip_features = second_clip_features.detach()
                                clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                # clip_features = clip_features.detach()
                    elif self.clip_method=='avg_frame':
                        single_frame = imgs[i][j]
                        if i == 0 and j == 4:# in 8 frame setting, 4th frame is the center frame
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                        elif i !=0 and j == 4:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                    else:
                        single_frame = imgs[i][j]
                        if i == 0 and j == 0:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # print('clip_features.shape 11111', clip_features.shape)
                        elif i==0 and j!=0 and j!=7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                #second_clip_features = second_clip_features.detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                        elif i==0 and j==7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                #second_clip_features = second_clip_features.detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                            # print('clip_features.shape ^88888^^^^^^ ', clip_features.shape)
                            clip_features = torch.mean(clip_features, 0,keepdim=True)
                            # print('clip_features.shape ^^^^^^^ ', clip_features.shape)
                            # clip_features = clip_features.detach()
                        else:
                            for a in range(1, imgs.shape[0]):
                                if i==a and j==0:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                elif i==a and j!=0 and j!=7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features = torch.cat((second_clip_features, third_clip_features), 0).detach()
                                elif i==a and j==7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features = torch.cat((second_clip_features, third_clip_features), 0).detach()
                                    # print('before mean second_clip_features', second_clip_features.shape)
                                    second_clip_features = torch.mean(second_clip_features, 0).unsqueeze(0)
                                    # print('after mean second_clip_features', second_clip_features.shape)
                                    
                                    # print('second_clip_features.shape ', second_clip_features.shape)
                                    clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                    # print('clip_features.shape ', clip_features.shape)

            print('clip_features - ', clip_features.shape)
       #------------------------------------------------------------------------ 
            t_clip_emb = clip_features.detach()
            t_embs = []
            t_embs.append(t_clip_emb)
            self.teacher_student = []
        # print('imgs that goes in to teacher', imgs.shape)
        # img size is torch.Size([12, 8, 3, 224, 224])
        # with torch.no_grad():
            if self.speed_network:
                self.teacher_student.append('speed')
                t_speed_emb = self.speed_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                # print('teacher speed')
                # print(t_speed_emb.shape)
                t_embs.append(t_speed_emb)
            if self.color_network:
                self.teacher_student.append('color')
                t_color_emb = self.color_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                # print('teacher color')
                # print(t_color_emb.shape)
                t_embs.append(t_color_emb)
            if self.vcop_network:
                self.teacher_student.append('vcop')
                t_vcop_emb = self.vcop_network.forward_teacher(imgs, emb_stage='backbone').detach()
                # print('vcop teacher')
                # print(t_vcop_emb.shape)
                t_embs.append(t_vcop_emb) # the output feature from the vcop trained tsm not head .... 
        
        # print("t_embs[0] --- teacher embedding size", t_embs[0].shape)

        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # imgs reshape to (batches*frames, c, h, w)???
        num_segs = imgs.shape[0] // batches

        losses = dict()
        loss_emb= dict() # loss for the student

        x = self.extract_feat(imgs) # is this tsm backbone that aare are extracting the features from ???
        print('x = self.extract_feat(imgs) ---- this goes into the student ', x.shape)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))
        x = nn.AdaptiveAvgPool2d(1)(x)
        print('student size', x.shape)
        x = x.squeeze()
        # print('student size', x.shape)
        # print('student  is above with nn.AdaptiveAvgPool2d(1)(x), x.squeeze()')
        # print('x = x.squeeze() -  this goes in to the contrastive head',x.shape)
        s_embs = []
        if clip_features.shape==x.shape:
            s_embs.append(x) # append since we already have clip feature in the t_embs
        else:# x 96, 2048 
            # print('x.shape', x.shape)
            for i in range(1, x.shape[0]+1):
                if i !=1 and i==8:
                    avg_x = torch.mean(x[i-8:i],0,keepdim=True)
                    # print('avg_x.shape', avg_x.shape)
                elif i!=1 and i!=8 and i%8==0:
                    avg_x_second =  torch.mean(x[i-8:i], 0, keepdim=True)
                    # print('avg_x_second.shape', avg_x_second.shape)
                    avg_x = torch.cat((avg_x, avg_x_second), 0)
                    # print('avg_x.shape1111', avg_x.shape)
            s_embs.append(avg_x)
            if avg_x.shape!= clip_features.shape:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print('error - -- - - - - - - -')
        if self.speed_network:
            # s_speed_emb = self.speed_contrastive_emb_head(x.float())
            # s_embs.append(s_speed_emb)
            # print(s_speed_emb.shape)
            s_speed_emb=x
            s_embs.append(s_speed_emb)
        if self.color_network:
            s_color_emb = x #self.color_contrastive_emb_head(x.float())
            #print(s_color_emb.shape)
            s_embs.append(s_color_emb)
        if self.vcop_network:
            s_vcop_emb =x
            s_embs.append(s_vcop_emb)

        # print('s_embs[0] ---  student embedding shape ', s_embs[0].shape)
        
        c1 = s_embs[0].detach().cpu().numpy()
        c2 = t_embs[0].detach().cpu().numpy()
        if np.all(c1==c2):
            print('student and teacher embeddings are same')
        # if t_embs==s_embs:
        #     print('same')
        # print(torch.equal(t_embs, s_embs))

        if self.type_loss == 'stack':
            t_embs = torch.vstack(t_embs)
            s_embs = torch.vstack(s_embs)
            loss_emb = self.emb_loss(t_embs, s_embs) # stacking the features
        elif self.type_loss == 'cat':
            layer_concat = torch.concat(t_embs, dim=1).to(device).float()
            concat_layer_size = layer_concat.shape[1]
            student_dim_size = x.shape[1]
            
            reduction_layer = nn.Linear(concat_layer_size, student_dim_size).to(device).float()
            print('running before linaer layer ---------------')
            t_embs = reduction_layer(layer_concat).to(device).float()
            print('passed the layer ~~~~~~~~~~~~~~~~~')
            s_embs=x.to(device).float()
            loss_emb = self.emb_loss(t_embs.detach(), s_embs)
        elif self.type_loss == 'avg':
            t_embs = torch.stack(t_embs)
            s_embs = torch.stack(s_embs)
            t_embs = torch.mean(t_embs, dim=0)
            s_embs = torch.mean(s_embs, dim=0)
            loss_emb = self.emb_loss(t_embs, s_embs)
        elif self.type_loss == 'random':
            model_random_list = torch.randint(0,4,size=(1,batches))
            model_random_list = model_random_list.tolist()
            model_random_list=model_random_list[0]
            all_batch = []
            for batch_num in range(batches):
                model_random_num = model_random_list[batch_num]
                teacher_model=t_embs[model_random_num]
                start_idx = batch_num*8
                one_batch = []
                for y in range(8):
                    idx = start_idx+y
                    one_batch.append(teacher_model[idx].unsqueeze(0))
                one_batch = torch.vstack(one_batch)
                all_batch.append(one_batch)
            t_embs = torch.vstack(all_batch)
            s_embs = x
            loss_emb = self.emb_loss(t_embs, s_embs)

        cls_score = self.cls_head(x.float(), num_segs)
        # cls_score = self.cls_head(x.to(torch.float32), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        losses.update(loss_emb)

        #losses.update(loss_cls)
        
        return losses








@RECOGNIZERS.register_module()
class MultipleContrastiveDistillerRecognizer_w_clip_speed(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 speed_network=None,
                 color_network=None,
                 vcop_network=None,
                 type_loss='stack',
                 contrastive_head=None,
                 emb_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 domain=None,
                 clip_method=None,
                 emb_stage='proj_head'):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.clip_method = clip_method
        self.emb_stage = emb_stage
        self.type_loss=type_loss
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

        normal_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D1_test_D1/best_top1_acc_epoch_85.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D3_test_D3/best_top1_acc_epoch_75.pth'}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("RN50", device='cpu')
        self.clip_model.visual.attnpool.c_proj = torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.clip_model.to(device)
        #self.clip_model.half()
        self.clip_model.float()

        if speed_network != None:

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
           # speed_ckpt_path="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1/tsm-k400-speed-contrastive_xd_sgd_speed_temp_5/train_{domain}_test_{domain}/latest.pth"
            self.speed_network = builder.build_model(speed_network)
            CPS_2_C_network_state_dict = torch.load(speed_ckpt_path[domain])
            print(speed_ckpt_path[domain])
            self.speed_network.load_state_dict(CPS_2_C_network_state_dict["state_dict"], strict=False)
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

        self.emb_loss = builder.build_loss(emb_loss)

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
        # tensor_to_img = T.ToPILImage()
        
        #--------------------------------------------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
   
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                for j in range(imgs.shape[1]):

                    if self.clip_method=='all':
                        single_frame = imgs[i][j]
                        if i == 0 and j == 0:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # clip_features = clip_features.detach()
                        else:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            #second_clip_features = second_clip_features.detach()
                            clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                            # clip_features = clip_features.detach()

                    else: # inflated center frame feature 
                        single_frame = imgs[i][j]
                        if i == 0 and j == 4:# in 8 frame setting, 4th frame is the center frame
                            clip_image = single_frame.unsqueeze(0).to(device)
                            clip_features=self.clip_model.encode_image(clip_image).detach()
                            # clip_features = clip_features.detach()
                            for a in range(imgs.shape[1]-1):
                                clip_image = single_frame.unsqueeze(0).to(device)
                                second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                #second_clip_features = second_clip_features.detach()
                                clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                # clip_features = clip_features.detach()
                        elif i !=0 and j == 4:
                            for a in range(imgs.shape[1]):
                                clip_image = single_frame.unsqueeze(0).to(device)
                                second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                # second_clip_features = second_clip_features.detach()
                                clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
                                # clip_features = clip_features.detach()

            print('clip_features - ', clip_features.shape)
       #------------------------------------------------------------------------ 
            t_clip_emb = clip_features.detach()
            t_embs = []
            t_embs.append(t_clip_emb)
            self.teacher_student = []
        # print('imgs that goes in to teacher', imgs.shape)
        # img size is torch.Size([12, 8, 3, 224, 224])
        # with torch.no_grad():
            if self.speed_network:
                self.teacher_student.append('speed')
                t_speed_emb = self.speed_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                # print('teacher speed')
                # print(t_speed_emb.shape)
                t_embs.append(t_speed_emb)
            if self.color_network:
                self.teacher_student.append('color')
                t_color_emb = self.color_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                # print('teacher color')
                # print(t_color_emb.shape)
                t_embs.append(t_color_emb)
            if self.vcop_network:
                self.teacher_student.append('vcop')
                t_vcop_emb = self.vcop_network.forward_teacher(imgs, emb_stage='backbone').detach()
                # print('vcop teacher')
                # print(t_vcop_emb.shape)
                t_embs.append(t_vcop_emb) # the output feature from the vcop trained tsm not head .... 
        
        # print("t_embs[0] --- teacher embedding size", t_embs[0].shape)

        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:]) # imgs reshape to (batches*frames, c, h, w)???
        num_segs = imgs.shape[0] // batches

        losses = dict()
        loss_emb= dict() # loss for the student

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
        #print('student size', x.shape)
        # print('student  is above with nn.AdaptiveAvgPool2d(1)(x), x.squeeze()')
        # print('x = x.squeeze() -  this goes in to the contrastive head',x.shape)
        s_embs = []
        s_embs.append(x) # append since we already have clip feature in the t_embs
        if self.speed_network:
            # s_speed_emb = self.speed_contrastive_emb_head(x.float())
            # s_embs.append(s_speed_emb)
            # print(s_speed_emb.shape)
            s_speed_emb=x
            s_embs.append(s_speed_emb)
        if self.color_network:
            s_color_emb = x #self.color_contrastive_emb_head(x.float())
            #print(s_color_emb.shape)
            s_embs.append(s_color_emb)
        if self.vcop_network:
            s_vcop_emb =x
            s_embs.append(s_vcop_emb)

        # print('s_embs[0] ---  student embedding shape ', s_embs[0].shape)

        c1 = s_embs[0].detach().cpu().numpy()
        c2 = t_embs[0].detach().cpu().numpy()
        if np.all(c1==c2):
            print('student and teacher embeddings are same')
        # if t_embs==s_embs:
        #     print('same')
        # print(torch.equal(t_embs, s_embs))
        if self.type_loss == 'stack':
            t_embs = torch.vstack(t_embs)
            s_embs = torch.vstack(s_embs)
            loss_emb = self.emb_loss(t_embs, s_embs)
        # else:
        #     if 'vcop' in self.teacher_student:
        #         vcop_loss_emb = self.emb_loss(t_vcop_emb, s_vcop_emb)
        #         loss_emb['loss_embedding']=vcop_loss_emb['loss_embedding']
        #     if 'color' in self.teacher_student:
        #         color_loss_emb = self.emb_loss(t_color_emb, s_color_emb)
        #         loss_emb['loss_embedding']=color_loss_emb['loss_embedding'] + loss_emb['loss_embedding']
        #     if 'speed' in self.teacher_student:
        #         speed_loss_emb = self.emb_loss(t_speed_emb, s_speed_emb)
        #         loss_emb['loss_embedding']=speed_loss_emb['loss_embedding'] + loss_emb['loss_embedding']

        #----------------------
            # loss_emb={}
            # color_loss_emb = self.emb_loss(t_color_emb, s_color_emb)
            # speed_loss_emb = self.emb_loss(t_speed_emb, s_speed_emb)
            # vcop_loss_emb = self.emb_loss(t_vcop_emb, s_vcop_emb)
            # loss_emb['loss_embedding']=color_loss_emb['loss_embedding']+speed_loss_emb['loss_embedding']+vcop_loss_emb['loss_embedding']
        #--------------------
        # print('t_embs --- teacher embedding shape ', t_embs.shape)
        # print('s_embs --- student embedding shape ', s_embs.shape)

        #loss_emb = self.emb_loss(t_embs, s_embs)
        cls_score = self.cls_head(x.float(), num_segs)
        # cls_score = self.cls_head(x.to(torch.float32), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)

        losses.update(loss_emb)

        #losses.update(loss_cls)
        
        return losses





