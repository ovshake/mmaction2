# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .. import builder
from einops import rearrange, reduce


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
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

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1
            losses.update(loss_aux)

        cls_score = self.cls_head(x.float(), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        x = nn.AdaptiveAvgPool2d(1)(x)
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

    def _do_fcn_test(self, imgs):
        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = self.test_cfg.get('num_segs', self.backbone.num_segments)

        if self.test_cfg.get('flip', False):
            imgs = torch.flip(imgs, [-1])
        x = self.extract_feat(imgs)

        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
        else:
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, fcn_test=True)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs, softmax=False):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if self.with_neck:
            x = [
                each.reshape((-1, num_segs) +
                             each.shape[1:]).transpose(1, 2).contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        outs = self.cls_head(x, num_segs)
        if softmax:
            outs = nn.functional.softmax(outs)
        return (outs, )

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        assert self.with_cls_head
        return self._do_test(imgs)



@RECOGNIZERS.register_module()
class SlowFastSelfSupervisedRecognizer2D(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 self_supervised_loss=None,
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
        self.self_supervised_loss = builder.build_loss(self_supervised_loss)


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

            imgs_slow, imgs_fast = imgs
            return self.forward_train(imgs_slow, imgs_fast, label, **kwargs)

        return self.forward_test(imgs, **kwargs)


    def forward_train(self, imgs_slow, imgs_fast, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        batches = imgs_slow.shape[0]
        imgs_slow = imgs_slow.reshape((-1, ) + imgs_slow.shape[2:])
        imgs_fast = imgs_fast.reshape((-1, ) + imgs_fast.shape[2:])
        num_segs = imgs_slow.shape[0] // batches

        losses = dict()

        x_slow = self.extract_feat(imgs_slow)
        x_fast = self.extract_feat(imgs_fast)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x_slow.shape) == 4 and (x_slow.shape[2] > 1 or x_slow.shape[3] > 1):
                # apply adaptive avg pooling
                x_slow = nn.AdaptiveAvgPool2d(1)(x_slow)

            if len(x_fast.shape) == 4 and (x_fast.shape[2] > 1 or x_fast.shape[3] > 1):
                # apply adaptive avg pooling
                x_fast = nn.AdaptiveAvgPool2d(1)(x_fast)

            x_slow = x_slow.reshape((x_slow.shape[0], -1))
            x_slow = x_slow.reshape(x_slow.shape + (1, 1))

            x_fast = x_fast.reshape((x_fast.shape[0], -1))
            x_fast = x_fast.reshape(x_fast.shape + (1, 1))

        # if self.with_neck:
        #     x = [
        #         each.reshape((-1, num_segs) +
        #                      each.shape[1:]).transpose(1, 2).contiguous()
        #         for each in x
        #     ]
        #     x, loss_aux = self.neck(x, labels.squeeze())
        #     x = x.squeeze(2)
        #     num_segs = 1
        #     losses.update(loss_aux)

        cls_score_slow = self.cls_head(x_slow, num_segs)
        cls_score_fast = self.cls_head(x_fast, num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score_slow, gt_labels, **kwargs)
        loss_self_supervised = self.self_supervised_loss(cls_score_slow, cls_score_fast)
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

@RECOGNIZERS.register_module()
class SlowFastSelfSupervisedContrastiveHeadRecognizer2D(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 slow_contrastive_head=None,
                 fast_contrastive_head=None,
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
        self.slow_contrastive_head = builder.build_head(slow_contrastive_head)
        self.fast_contrastive_head = builder.build_head(fast_contrastive_head)
        self.slow_fast_contrastive_loss = builder.build_loss(contrastive_loss)
        self.fast_to_slow_projection_head = nn.Linear(self.fast_contrastive_head.img_dim,
                                                self.slow_contrastive_head.img_dim, bias=True)

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

            imgs_slow, imgs_fast = imgs
            return self.forward_train(imgs_slow, imgs_fast, label, **kwargs)

        return self.forward_test(imgs, **kwargs)


    def forward_train(self, imgs_slow, imgs_fast, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        batches = imgs_slow.shape[0]
        imgs_slow = imgs_slow.reshape((-1, ) + imgs_slow.shape[2:])
        imgs_fast = imgs_fast.reshape((-1, ) + imgs_fast.shape[2:])
        num_segs = imgs_slow.shape[0] // batches

        losses = dict()

        x_slow = self.extract_feat(imgs_slow)
        x_fast = self.extract_feat(imgs_fast)
        x_slow = nn.AdaptiveAvgPool2d(1)(x_slow)
        x_fast = nn.AdaptiveAvgPool2d(1)(x_fast)
        x_slow = x_slow.squeeze()
        x_fast = x_fast.squeeze()
        contrastive_slow_features = self.slow_contrastive_head(x_slow.float())
        contrastive_fast_features = self.fast_contrastive_head(x_fast.float())
        projected_fast_features = self.fast_to_slow_projection_head(contrastive_fast_features)


        cls_score_fast = self.cls_head(x_fast.float(), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score_fast, gt_labels, **kwargs)
        loss_self_supervised = self.slow_fast_contrastive_loss(contrastive_slow_features, projected_fast_features)
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


@RECOGNIZERS.register_module()
class VCOPSRecognizer2D(Recognizer2D):
    """2D recognizer model framework for Video Clip Order Prediction"""
    def __init__(self,
                 backbone,
                 cls_head=None,
                 vcop_head=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_clips=1):
        super().__init__(backbone,
                        cls_head=cls_head,
                        neck=neck,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg)

        self.num_clips = num_clips
        self.vcop_head = builder.build_head(vcop_head)

    def forward_train(self, imgs, labels, **kwargs):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert self.vcop_head
        batches = imgs.shape[0]
        num_frames = imgs.shape[1] // self.num_clips
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze()
        vcop_loss = self.vcop_head(x.reshape(batches, self.num_clips, num_segs // self.num_clips, -1).float(),
                                   return_loss=True)

        # x = x.reshape((batches, self.num_clips, num_frames,-1))
        cls_score = self.cls_head(x.float(), num_segs)

        # num_batch x num_clips x num_classes
        cls_score = cls_score.view(batches, -1, cls_score.size()[-1])
        # num_batch x 1 x num_classes
        cls_score = self.cls_head.consensus(cls_score)

        # num_batch x num_classes
        cls_score = cls_score.squeeze(1)

        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)
        losses.update(vcop_loss)
        return losses


@RECOGNIZERS.register_module()
class ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 vanilla_contrastive_head=None,
                 color_contrastive_head=None,
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

        self.vanilla_contrastive_head = builder.build_head(vanilla_contrastive_head)
        self.color_contrastive_head = builder.build_head(color_contrastive_head)
        self.color_to_vanilla_projection_layer = nn.Linear(self.color_contrastive_head.img_dim,
                                                self.vanilla_contrastive_head.img_dim, bias=True)


        self.color_contrastive_loss = builder.build_loss(contrastive_loss)
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
        x_pathway_B = self.extract_feat(imgs_pathway_B)
        x_pathway_A = nn.AdaptiveAvgPool2d(1)(x_pathway_A)
        x_pathway_B = nn.AdaptiveAvgPool2d(1)(x_pathway_B)
        x_pathway_A = x_pathway_A.squeeze()
        x_pathway_B = x_pathway_B.squeeze()

        contrastive_pathway_A_features = self.vanilla_contrastive_head(x_pathway_A.float())
        contrastive_pathway_B_features = self.color_contrastive_head(x_pathway_B.float())
        proj_contrastive_pathway_B_features = self.color_to_vanilla_projection_layer(contrastive_pathway_B_features)

        if self.backbone_from in ['torchvision', 'timm']:
            if len(x_pathway_A.shape) == 4 and (x_pathway_A.shape[2] > 1 or x_pathway_A.shape[3] > 1):
                # apply adaptive avg pooling
                x_pathway_A = nn.AdaptiveAvgPool2d(1)(x_pathway_A)

            if len(x_pathway_B.shape) == 4 and (x_pathway_B.shape[2] > 1 or x_pathway_B.shape[3] > 1):
                # apply adaptive avg pooling
                x_pathway_B = nn.AdaptiveAvgPool2d(1)(x_pathway_B)

            x_pathway_A = x_pathway_A.reshape((x_pathway_A.shape[0], -1))
            x_pathway_A = x_pathway_A.reshape(x_pathway_A.shape + (1, 1))

            x_pathway_B = x_pathway_B.reshape((x_pathway_B.shape[0], -1))
            x_pathway_B = x_pathway_B.reshape(x_pathway_B.shape + (1, 1))

        cls_score_pathway_A = self.cls_head(x_pathway_A.float(), num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score_pathway_A, gt_labels, **kwargs)
        loss_self_supervised = self.color_contrastive_loss(contrastive_pathway_A_features, proj_contrastive_pathway_B_features)
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

@RECOGNIZERS.register_module()
class MultipleContrastiveRecognizer2D(Recognizer2D):
    def __init__(self, backbone,
                 cls_head=None,
                 num_contrastive_heads=None,
                 contrastive_head=None,
                 self_supervised_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_segments=None,
                 device='cuda'):

        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)

        # self.cls_head = builder.build_head(cls_head) if cls_head else None
        self.self_supervised_loss = builder.build_loss(self_supervised_loss)

        assert num_contrastive_heads, 'Need atleast 1 contrastive head!'
        assert contrastive_head, 'Need configuration of the contrastive head!'

        self.contrastive_heads = [builder.build_head(contrastive_head).to(device) for _ in range(num_contrastive_heads)]


    def process_pathways(self, imgs):
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        x = self.extract_feat(imgs)
        if self.backbone_from in ['torchvision', 'timm']:
            if len(x.shape) == 4 and (x.shape[2] > 1 or x.shape[3] > 1):
                x = nn.AdaptiveAvgPool2d(1)(x)

            x = x.reshape((x.shape[0], -1))
            x = x.reshape(x.shape + (1, 1))

        elif self.backbone_from in ['mmaction2']:
            if len(x.shape) == 4 and x.shape[1] > 1:
                x = nn.AdaptiveAvgPool2d(1)(x)
                x = x.squeeze()

        return x

    def forward_train(self, imgs_pathways, labels, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        losses = dict()

        processed_pathways = []
        for imgs in imgs_pathways:
            x = self.process_pathways(imgs.float())
            processed_pathways.append(x)

        embedding_spaces = None


        for idx in range(len(self.contrastive_heads)):

            img_embeddings = None
            for feature_img in processed_pathways:
                h_img = self.contrastive_heads[idx](feature_img.float())
                if img_embeddings is None:
                    img_embeddings = h_img.unsqueeze(0)
                else:
                    img_embeddings = torch.vstack((img_embeddings, h_img.unsqueeze(0)))

            if embedding_spaces is None:
                embedding_spaces = img_embeddings.unsqueeze(0)
            else:
                embedding_spaces = torch.vstack((embedding_spaces, img_embeddings.unsqueeze(0)))


        # For classification I am taking features of Q from all the embedding space
        # and feeding them to the classifier.

        cls_features = processed_pathways[-1]
        del processed_pathways
        cls_score = self.cls_head(cls_features.float(), -1)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        loss_self_supervised = self.self_supervised_loss(embedding_spaces)
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
        num_pathways = len(data_batch)
        imgs = []
        for i in range(num_pathways):
            imgs.append(data_batch[i]['imgs'])

        label = data_batch[0]['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(imgs, label=label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch[0].values()))))

        return outputs

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        x = self.extract_feat(imgs.float())
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze()
        losses = dict()

        cls_score = self.cls_head(x.float(), num_segs)
        cls_score = self.average_clip(cls_score,
                                    cls_score.size()[0] // batches)
        return cls_score
