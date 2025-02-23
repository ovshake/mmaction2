# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .. import builder
from einops import rearrange
from .recognizer2d import Recognizer2D
from torch.nn.functional import normalize
import torch.distributed as dist
from .color_contrastive_recognizer2d import ColorSpatialSelfSupervised1SimSiamContrastiveHeadRecognizer2D


@RECOGNIZERS.register_module()
class MultipleContrastiveDistillerRecognizer(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 speed_network=None,
                 color_network=None,
                 contrastive_head=None,
                 emb_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 domain=None,
                 emb_stage='proj_head'):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)

        self.emb_stage = emb_stage
        if speed_network:
            self.speed_network = builder.build_model(speed_network).eval()
        else:
            self.speed_network = None

        if color_network:
            self.color_network = builder.build_model(color_network).eval()
        else:
            self.color_network = None

        speed_network_ckpt_dict = {
            "D1": "/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v1/speed-contrastive-simsiam-xdb/train_D1_test_D1/best_top1_acc_epoch_45.pth",
            "D2": "/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v1/speed-contrastive-simsiam-xdb/train_D2_test_D2/best_top1_acc_epoch_30.pth",
            "D3": "/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v1/speed-contrastive-simsiam-xdb/train_D3_test_D3/best_top1_acc_epoch_65.pth"
        }

        color_network_ckpt_dict = {
            "D1": "/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_cont_ablation/color-contrastive-single-instance-X-b-d-mean-simsiam-proj-layer/train_D1_test_D1/best_top1_acc_epoch_45.pth",
            "D2": "/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_cont_ablation/color-contrastive-single-instance-X-b-d-mean-simsiam-proj-layer/train_D2_test_D2/best_top1_acc_epoch_45.pth",
            "D3": "/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_cont_ablation/color-contrastive-single-instance-X-b-d-mean-simsiam-proj-layer/train_D3_test_D3/best_top1_acc_epoch_50.pth",
        }


        if self.speed_network:
            self.speed_network.load_state_dict(torch.load(speed_network_ckpt_dict[domain]), strict=False)
            self.speed_contrastive_emb_head = builder.build_head(contrastive_head)
        if self.color_network:
            self.color_network.load_state_dict(torch.load(color_network_ckpt_dict[domain]), strict=False)
            self.color_contrastive_emb_head = builder.build_head(contrastive_head)

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

        return self.forward_test(imgs, **kwargs)


    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        t_embs = []
        with torch.no_grad():
            if self.speed_network:
                t_speed_emb = self.speed_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                t_embs.append(t_speed_emb)
            if self.color_network:
                t_color_emb = self.color_network.forward_teacher(imgs, emb_stage=self.emb_stage).detach()
                t_embs.append(t_color_emb)

        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze()
        s_embs = []
        if self.speed_network:
            s_speed_emb = self.speed_contrastive_emb_head(x.float())
            s_embs.append(s_speed_emb)
        if self.color_network:
            s_color_emb = self.color_contrastive_emb_head(x.float())
            s_embs.append(s_color_emb)

        t_embs = torch.vstack(t_embs)
        s_embs = torch.vstack(s_embs)

        loss_emb = self.emb_loss(t_embs, s_embs)
        cls_score = self.cls_head(x.float(), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)
        losses.update(loss_emb)
        return losses






