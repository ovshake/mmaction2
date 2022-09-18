# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .. import builder
from einops import rearrange
from .recognizer2d import Recognizer2D
from torch.nn.functional import normalize


class TemporalPool(nn.Module):
        """Temporal pool module.

        Wrap layer2 in ResNet50 with a 3D max pooling layer.

        Args:
            net (nn.Module): Module to make temporal pool.
            num_segments (int): Number of frame segments.
        """

        def __init__(self, num_segments):
            super().__init__()
            self.num_segments = num_segments
            self.max_pool3d = nn.MaxPool3d(
                kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))

        def forward(self, x):
            # [N, C, H, W]
            n, c, h, w = x.size()
            # [N // num_segments, C, num_segments, H, W]
            x = x.view(n // self.num_segments, self.num_segments, c, h,
                        w).transpose(1, 2)
            # [N // num_segmnets, C, num_segments // 2, H, W]
            x = self.max_pool3d(x)
            # [N // 2, C, H, W]
            x = x.transpose(1, 2).contiguous().view(n // 2, c, h, w)
            return x

@RECOGNIZERS.register_module()
class TwoPathwaySelfSupervised1SimSiamCosSimRecognizer2D(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 contrastive_head=None,
                 contrastive_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        # record the source of the backbone
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

        self.contrastive_head = builder.build_head(contrastive_head)


        self.contrastive_loss = builder.build_loss(contrastive_loss)
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

        self.fp16_enabled = False
        self.A_to_B_projection_layer = nn.Linear(self.contrastive_head.img_dim,
                                                self.contrastive_head.img_dim, bias=True)


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


    def forward_train(self, imgs_pathway_A, imgs_pathway_B, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        batches = imgs_pathway_A.shape[0]
        imgs_pathway_A = imgs_pathway_A.reshape((-1, ) + imgs_pathway_A.shape[2:])
        imgs_pathway_B = imgs_pathway_B.reshape((-1, ) + imgs_pathway_B.shape[2:])
        num_segs = imgs_pathway_A.shape[0] // batches

        losses = dict()

        x_pathway_A = self.extract_feat(imgs_pathway_A)
        cls_x_pathway_A = x_pathway_A.clone()
        x_pathway_B = self.extract_feat(imgs_pathway_B)
        x_pathway_A = nn.AdaptiveAvgPool2d(1)(x_pathway_A)
        x_pathway_B = nn.AdaptiveAvgPool2d(1)(x_pathway_B)
        x_pathway_A = x_pathway_A.squeeze()
        x_pathway_B = x_pathway_B.squeeze()

        contrastive_pathway_A_features = self.contrastive_head(x_pathway_A.float())
        contrastive_pathway_B_features = self.contrastive_head(x_pathway_B.float())

        cls_x_pathway_A = nn.AdaptiveAvgPool2d(1)(cls_x_pathway_A)
        cls_x_pathway_A = cls_x_pathway_A.squeeze()
        cls_score_pathway_A = self.cls_head(cls_x_pathway_A.float(), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score_pathway_A, gt_labels, **kwargs)
        proj_features_A_features = self.A_to_B_projection_layer(contrastive_pathway_A_features)

        loss_self_supervised = self.contrastive_loss(proj_features_A_features,
                                        contrastive_pathway_B_features.detach())

        losses.update(loss_cls)
        losses.update(loss_self_supervised)
        return losses


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

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze()

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                # apply adaptive avg pooling
                x = nn.AdaptiveAvgPool2d(1)(x)
            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        if self.feature_extraction:
            # perform spatial pooling
            avg_pool = nn.AdaptiveAvgPool2d(1)
            x = avg_pool(x)
            # squeeze dimensions
            x = x.reshape((batches, num_segs, -1))
            # temporal average pooling
            x = x.mean(axis=1)
            return x

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`

        # should have cls_head if not extracting features
        cls_score = self.cls_head(x.float(), num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

