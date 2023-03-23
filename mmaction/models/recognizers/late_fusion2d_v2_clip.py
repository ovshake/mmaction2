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
import clip
from PIL import Image
import torchvision.transforms as T
from .recognizer2d import Recognizer2D

@RECOGNIZERS.register_module()
class LateFusionRecognizer_all_in_one_clip(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 CPS_network=None,
                 color_network=None,
                 normal_network=None,
                 vcop_network=None,
                 domain=None,
                 train_cfg=None,
                 test_cfg=None,
                 clip_method='all',
                 reduction=False,
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.CPS_network=CPS_network
        self.color_network=color_network
        self.vcop_network=vcop_network
        self.normal_network=normal_network
        self.clip_method = clip_method
        self.reduction = reduction
        print('*******************************')
        print(domain, '<<<<<=====domain')
        print('*******************************')
        
        

        
        vcop_ckpt_path = {'D1': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D1_test_D1/best_top1_acc_epoch_80.pth',
                            'D2': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_vcop_cls/tsm-k400-vcop_clip4_batch_12_cls/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        CPS_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D1_test_D1/best_top1_acc_epoch_90.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        Color_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D1_test_D1/best_top1_acc_epoch_55.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_speed_pathway/train_D3_test_D3/best_top1_acc_epoch_60.pth'}

        Normal_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D1_test_D1/best_top1_acc_epoch_85.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D3_test_D3/best_top1_acc_epoch_75.pth'}

#    CPS_network=None,
#                  color_network=None,
#                  normal_network=None,
#                  vcop_network=None,
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("RN50", device='cpu')
        self.clip_model.visual.attnpool.c_proj = torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.clip_model.to(device)
        self.clip_model.half()

        self.idx = 1
        if CPS_network:
            self.idx +=1

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
            self.CPS_network = builder.build_model(CPS_network)
            speed_network_state_dict = torch.load(CPS_ckpt_path[domain])
            self.CPS_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        else:
            self.CPS_network = None
        if color_network:
            self.idx +=1
            
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(Color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        else:
            self.color_network = None
        
        if normal_network:
            self.normal_network = None
            # self.idx +=1
            
            # self.normal_network = builder.build_model(normal_network)
            # normal_network_state_dict = torch.load(Normal_ckpt_path[domain])
            # self.color_network.load_state_dict(normal_network_state_dict["state_dict"], strict=False)
        else: 
            self.normal_network = None
        if vcop_network:
            self.idx +=1
            
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
            # vcop_head = dict(type='VCOPHead',num_clips=3, feature_size=2048*7*7 )
            # self.vcop_emb_head = builder.build_head(vcop_head)
        else:
            self.vcop_network = None

        if self.reduction:
            in_channel_size = 2048
        else:
            in_channel_size = 2048*self.idx
        cls_head=dict(type='TSMHead',
                        num_segments=8,
                        num_classes=8,
                        spatial_type=None,
                        consensus=dict(type='AvgConsensus', dim=1),
                        in_channels=in_channel_size,
                        init_std=0.001,
                        dropout_ratio=0.0)

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
 
        if self.idx==3:
            imgs_slow, imgs_fast = data_batch[0]['imgs'], data_batch[1]['imgs']

            label = data_batch[0]['label']
            losses = self((imgs_slow,imgs_fast, imgs_fast), label, return_loss=True)
        elif self.idx==4:
            imgs_slow, imgs_fast,img_mid = data_batch[0]['imgs'], data_batch[1]['imgs'],data_batch[2]['imgs']
  
            label = data_batch[0]['label']
            losses = self((imgs_slow, imgs_fast,imgs_fast,img_mid), label, return_loss=True)
        elif self.idx ==5:
          
            imgs_slow, imgs_fast,img_mid,img_vcop = data_batch[0]['imgs'], data_batch[1]['imgs'],data_batch[2]['imgs'],data_batch[3]['imgs']
            label = data_batch[0]['label']
            losses = self((imgs_slow,imgs_slow, imgs_fast,img_mid,img_vcop), label, return_loss=True)     


        # losses = self((imgs_slow, imgs_fast,img_mid), label, return_loss=True)

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
            if self.idx ==2:
                imgs_pathway_A, imgs_pathway_B = imgs
            elif self.idx ==3:
                imgs_pathway_A, imgs_pathway_B,imgs_pathway_C = imgs
            elif self.idx ==4:
                imgs_pathway_A, imgs_pathway_B,imgs_pathway_C,imgs_pathway_D = imgs
    
            return self.forward_train(imgs_pathway_A, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, imgs, gt_labels, **kwargs):
        assert self.with_cls_head
       
        #---------------------
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
        #---------------------
        clip_features = clip_features.detach()
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        ensemble_network = []
        ensemble_network.append(clip_features)
        if self.CPS_network:
            cps_features = self.CPS_network.extract_feat(imgs)
            cps_features = nn.AdaptiveAvgPool2d(1)(cps_features).squeeze()
            ensemble_network.append(cps_features)
        else:
            cps_features = None
        if self.color_network:
            color_features = self.color_network.extract_feat(imgs)
            color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
            ensemble_network.append(color_features)
        else:
            color_features = None
        if self.normal_network:
            normal_features = self.normal_network.extract_feat(imgs)
            normal_features = nn.AdaptiveAvgPool2d(1)(normal_features).squeeze()
            ensemble_network.append(normal_features)
        else:
            normal_features = None
        if self.vcop_network:
            vcop_features = self.vcop_network.extract_feat(imgs)
            vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
            ensemble_network.append(vcop_features)
        else:
            vcop_features = None



        losses = dict()
        fused_features = torch.cat(ensemble_network,dim=1).to(device).float()
        if self.reduction:
            #layer_concat = torch.concat(t_embs, dim=1).to(device).float()
            concat_layer_size = fused_features.shape[1]
            reduction_layer = nn.Linear(concat_layer_size, 2048).to(device).float()
            fused_features = reduction_layer(fused_features).to(device).float()
         
        # else:
        # if len(ensemble_network) == 2:

        #     fused_features = torch.cat((ensemble_network[0], ensemble_network[1]), dim=1)
        # elif len(ensemble_network) == 3:
        #     fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2]), dim=1)
        # elif len(ensemble_network) == 4:
        #     fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2],ensemble_network[3]), dim=1)
        #fused_features = F.normalize(fused_features, dim=1, eps=1e-8) # normalizing the concatenated feature 
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        #---------------------
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
                        elif i==0 and j!=0 and j!=7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            clip_features+=second_clip_features
                        elif i==0 and j==7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            clip_features+=second_clip_features
                            clip_features/=8
                            # clip_features = clip_features.detach()
                        else:
                            for a in range(1, imgs.shape[1]):
                                if i==a and j==0:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                elif i==a and j!=0 and j!=7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features+=third_clip_features
                                elif i==a and j==7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features+=third_clip_features
                                    second_clip_features/=8
                                    clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
        #---------------------

        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        ensemble_network = []
        ensemble_network.append(clip_features)
        if self.CPS_network:
            cps_features = self.CPS_network.extract_feat(imgs)
            cps_features = nn.AdaptiveAvgPool2d(1)(cps_features).squeeze()
            ensemble_network.append(cps_features)
        else:
            cps_features = None
        if self.color_network:
            color_features = self.color_network.extract_feat(imgs)
            color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
            ensemble_network.append(color_features)
        else:
            color_features = None
        if self.normal_network:
            normal_features = self.normal_network.extract_feat(imgs)
            normal_features = nn.AdaptiveAvgPool2d(1)(normal_features).squeeze()
            ensemble_network.append(normal_features)
        else:
            normal_features = None
        if self.vcop_network:
            vcop_features = self.vcop_network.extract_feat(imgs)
            vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
            ensemble_network.append(vcop_features)
        else:
            vcop_features = None




        # if len(ensemble_network) == 2:

        #     fused_features = torch.cat((ensemble_network[0], ensemble_network[1]), dim=1)
        # elif len(ensemble_network) == 3:
        #     fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2]), dim=1)
        # elif len(ensemble_network) == 4:
        #     fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2],ensemble_network[3]), dim=1)
        fused_features = torch.cat(ensemble_network, dim=1).to(device).float()
        # color_features = F.normalize(color_features, dim=1, eps=1e-8)
        # speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        # vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)

        # fused_features = fused_features.norm(dim=None)
    
        # fused_features = fused_features.norm(dim=1)[:, None]
        if self.reduction:
            #layer_concat = torch.concat(t_embs, dim=1).to(device).float()
            concat_layer_size = fused_features.shape[1]
            # print('Print before the reduction layer ~~~~~~~~')
            reduction_layer = nn.Linear(concat_layer_size, 2048).to(device).float()
            # print('passed the reduction layer ~~~~~~~~')
            fused_features = reduction_layer(fused_features).to(device).float()

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score
#-----------------------------------------------------------------------

@RECOGNIZERS.register_module()
class LateFusionRecognizer_all_in_one_ucf_hmdb_clip(Recognizer2D):
    def __init__(self,
                 cls_head,
                 backbone,
                 CPS_network=None,
                 color_network=None,
                 normal_network=None,
                 vcop_network=None,
                 domain=None,
                 train_cfg=None,
                 test_cfg=None,
                 clip_method='all',
                 **kwargs):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.CPS_network=CPS_network
        self.color_network=color_network
        self.vcop_network=vcop_network
        self.normal_network=normal_network
        self.clip_method = clip_method
        print('*******************************')
        print(domain, '<<<<<=====domain')
        print('*******************************')
        
        

        
        vcop_ckpt_path = {'ucf101': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_UCF_VCOP_clip4_new/ucf/best_top1_acc_epoch_5.pth',
                            'hmdb51': '/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_HMDB_VCOP_clip4_new/hmdb/best_top1_acc_epoch_15.pth'}

        speed_ckpt_path = {'ucf101':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_UCF_two_pathway_CPS_C/ucf/best_top1_acc_epoch_70.pth',
                            'hmdb51':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_HMDB_two_pathway_CPS_C/hmdb/best_top1_acc_epoch_15.pth'}

        color_ckpt_path = {'ucf101':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_UCF_two_pathway_C_S/ucf/best_top1_acc_epoch_10.pth',
                            'hmdb51':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls/tsm_k400_HMDB_two_pathway_C_S/hmdb/best_top1_acc_epoch_15.pth'}

       

        Normal_ckpt_path = {'D1':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D1_test_D1/best_top1_acc_epoch_85.pth',
                            'D2':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D2_test_D2/best_top1_acc_epoch_55.pth',
                            'D3':'/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-normal_color_pathway/train_D3_test_D3/best_top1_acc_epoch_75.pth'}

#    CPS_network=None,
#                  color_network=None,
#                  normal_network=None,
#                  vcop_network=None,
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("RN50", device='cpu')
        self.clip_model.visual.attnpool.c_proj = torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.clip_model.to(device)
        self.clip_model.half()

        self.idx = 1
        if CPS_network:
            self.idx +=1

        #speed_ckpt_path = "/data/jongmin/projects/mmaction2/work_/speed/{domain}/latest.pth"
            self.CPS_network = builder.build_model(CPS_network)
            speed_network_state_dict = torch.load(CPS_ckpt_path[domain])
            self.CPS_network.load_state_dict(speed_network_state_dict["state_dict"], strict=False)
        else:
            self.CPS_network = None
        if color_network:
            self.idx +=1
            
            self.color_network = builder.build_model(color_network)
            color_network_state_dict = torch.load(Color_ckpt_path[domain])
            self.color_network.load_state_dict(color_network_state_dict["state_dict"], strict=False)
        else:
            self.color_network = None
        
        if normal_network:
            self.normal_network = None
            # self.idx +=1
            
            # self.normal_network = builder.build_model(normal_network)
            # normal_network_state_dict = torch.load(Normal_ckpt_path[domain])
            # self.color_network.load_state_dict(normal_network_state_dict["state_dict"], strict=False)
        else: 
            self.normal_network = None
        if vcop_network:
            self.idx +=1
            
            self.vcop_network = builder.build_model(vcop_network)
            vcop_network_state_dict = torch.load(vcop_ckpt_path[domain])
            self.vcop_network.load_state_dict(vcop_network_state_dict["state_dict"], strict=False)
            # vcop_head = dict(type='VCOPHead',num_clips=3, feature_size=2048*7*7 )
            # self.vcop_emb_head = builder.build_head(vcop_head)
        else:
            self.vcop_network = None


        cls_head=dict(type='TSMHead',
                        num_segments=8,
                        num_classes=12,
                        spatial_type=None,
                        consensus=dict(type='AvgConsensus', dim=1),
                        in_channels=2048*self.idx,
                        init_std=0.001,
                        dropout_ratio=0.0)

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
 
        if self.idx==3:
            imgs_slow, imgs_fast = data_batch[0]['imgs'], data_batch[1]['imgs']

            label = data_batch[0]['label']
            losses = self((imgs_slow,imgs_fast, imgs_fast), label, return_loss=True)
        elif self.idx==4:
            imgs_slow, imgs_fast,img_mid = data_batch[0]['imgs'], data_batch[1]['imgs'],data_batch[2]['imgs']
  
            label = data_batch[0]['label']
            losses = self((imgs_slow, imgs_fast,imgs_fast,img_mid), label, return_loss=True)
        elif self.idx ==5:
          
            imgs_slow, imgs_fast,img_mid,img_vcop = data_batch[0]['imgs'], data_batch[1]['imgs'],data_batch[2]['imgs'],data_batch[3]['imgs']
            label = data_batch[0]['label']
            losses = self((imgs_slow,imgs_slow, imgs_fast,img_mid,img_vcop), label, return_loss=True)     


        # losses = self((imgs_slow, imgs_fast,img_mid), label, return_loss=True)

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
            if self.idx ==2:
                imgs_pathway_A, imgs_pathway_B = imgs
            elif self.idx ==3:
                imgs_pathway_A, imgs_pathway_B,imgs_pathway_C = imgs
            elif self.idx ==4:
                imgs_pathway_A, imgs_pathway_B,imgs_pathway_C,imgs_pathway_D = imgs
    
            return self.forward_train(imgs_pathway_A, label, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, imgs, gt_labels, **kwargs):
        assert self.with_cls_head
       
        #---------------------
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
        #---------------------
        clip_features = clip_features.detach()
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        ensemble_network = []
        ensemble_network.append(clip_features)
        if self.CPS_network:
            cps_features = self.CPS_network.extract_feat(imgs)
            cps_features = nn.AdaptiveAvgPool2d(1)(cps_features).squeeze()
            ensemble_network.append(cps_features)
        else:
            cps_features = None
        if self.color_network:
            color_features = self.color_network.extract_feat(imgs)
            color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
            ensemble_network.append(color_features)
        else:
            color_features = None
        if self.normal_network:
            normal_features = self.normal_network.extract_feat(imgs)
            normal_features = nn.AdaptiveAvgPool2d(1)(normal_features).squeeze()
            ensemble_network.append(normal_features)
        else:
            normal_features = None
        if self.vcop_network:
            vcop_features = self.vcop_network.extract_feat(imgs)
            vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
            ensemble_network.append(vcop_features)
        else:
            vcop_features = None



        losses = dict()
        if len(ensemble_network) == 2:

            fused_features = torch.cat((ensemble_network[0], ensemble_network[1]), dim=1)
        elif len(ensemble_network) == 3:
            fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2]), dim=1)
        elif len(ensemble_network) == 4:
            fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2],ensemble_network[3]), dim=1)
        #fused_features = F.normalize(fused_features, dim=1, eps=1e-8) # normalizing the concatenated feature 
        cls_scores = self.cls_head(fused_features.float(), num_segs)
        gt_labels = gt_labels.squeeze()
        loss_cls = self.cls_head.loss(cls_scores, gt_labels, **kwargs)
        losses.update(loss_cls)
        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""

        device = "cuda" if torch.cuda.is_available() else "cpu"
        #---------------------
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
                        elif i==0 and j!=0 and j!=7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            clip_features+=second_clip_features
                        elif i==0 and j==7:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            second_clip_features = self.clip_model.encode_image(clip_image).detach()
                            clip_features+=second_clip_features
                            clip_features/=8
                            # clip_features = clip_features.detach()
                        else:
                            for a in range(1, imgs.shape[1]):
                                if i==a and j==0:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    second_clip_features = self.clip_model.encode_image(clip_image).detach()
                                elif i==a and j!=0 and j!=7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features+=third_clip_features
                                elif i==a and j==7:
                                    clip_image = single_frame.unsqueeze(0).to(device)
                                    third_clip_features = self.clip_model.encode_image(clip_image).detach()
                                    second_clip_features+=third_clip_features
                                    second_clip_features/=8
                                    clip_features = torch.cat((clip_features, second_clip_features), 0).detach()
        #---------------------

        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        ensemble_network = []
        ensemble_network.append(clip_features)
        if self.CPS_network:
            cps_features = self.CPS_network.extract_feat(imgs)
            cps_features = nn.AdaptiveAvgPool2d(1)(cps_features).squeeze()
            ensemble_network.append(cps_features)
        else:
            cps_features = None
        if self.color_network:
            color_features = self.color_network.extract_feat(imgs)
            color_features = nn.AdaptiveAvgPool2d(1)(color_features).squeeze()
            ensemble_network.append(color_features)
        else:
            color_features = None
        if self.normal_network:
            normal_features = self.normal_network.extract_feat(imgs)
            normal_features = nn.AdaptiveAvgPool2d(1)(normal_features).squeeze()
            ensemble_network.append(normal_features)
        else:
            normal_features = None
        if self.vcop_network:
            vcop_features = self.vcop_network.extract_feat(imgs)
            vcop_features = nn.AdaptiveAvgPool2d(1)(vcop_features).squeeze()
            ensemble_network.append(vcop_features)
        else:
            vcop_features = None




        if len(ensemble_network) == 2:

            fused_features = torch.cat((ensemble_network[0], ensemble_network[1]), dim=1)
        elif len(ensemble_network) == 3:
            fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2]), dim=1)
        elif len(ensemble_network) == 4:
            fused_features = torch.cat((ensemble_network[0], ensemble_network[1], ensemble_network[2],ensemble_network[3]), dim=1)

        # color_features = F.normalize(color_features, dim=1, eps=1e-8)
        # speed_features = F.normalize(speed_features, dim=1, eps=1e-8)
        # vcop_features = F.normalize(vcop_features, dim=1, eps=1e-8)

        # fused_features = fused_features.norm(dim=None)
    
        # fused_features = fused_features.norm(dim=1)[:, None]

        cls_score = self.cls_head(fused_features.float(), num_segs)
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score
#-----------------------------------------------------------------------

