# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer
from .. import builder
from einops import rearrange
from .recognizer2d import Recognizer2D
from torch.nn.functional import normalize



@RECOGNIZERS.register_module()
class MultiplePathwaySelfSupervised1SimSiamCosSimRecognizer2D(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 contrastive_head=None,
                 contrastive_loss=None,
                 neck=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_heads=None,
                 detach_pathway_num=None,
                 freeze_cls_head=False,
                 device='cuda'):

        super().__init__(backbone=backbone,
                        cls_head=cls_head,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg)

        self.backbone_from = 'mmaction2'
        torch.autograd.set_detect_anomaly(True)
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

        self.num_heads = num_heads
        self.detach_pathway_num = detach_pathway_num
        self.contrastive_heads = []
        for _ in range(num_heads):
            self.contrastive_heads.append(builder.build_head(contrastive_head).to(device))

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
        self.freeze_cls_head = freeze_cls_head
        # if self.freeze_cls_head:
        #     self.cls_head.eval()

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

    def process_pathways(self, imgs, **kwargs):
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.squeeze()
        return x

    def forward_train(self, imgs_pathways, labels, **kwargs):
        assert self.with_cls_head
        if self.freeze_cls_head:
            self.cls_head.eval()

        losses = dict()

        processed_pathways = []
        for imgs in imgs_pathways:
            x = self.process_pathways(imgs.float())
            processed_pathways.append(x)

        embedding_spaces = None
        labels = labels.squeeze()

        cls_scores = self.cls_head(processed_pathways[-1].float(), -1)
        loss_cls = self.cls_head.loss(cls_scores, labels, **kwargs)
        losses.update(loss_cls)
        for idx in range(len(self.contrastive_heads)):
            h_img = self.contrastive_heads[idx](processed_pathways[idx].float())

            if embedding_spaces is None:
                embedding_spaces = h_img.unsqueeze(0)
            else:
                embedding_spaces = torch.vstack((embedding_spaces, h_img.unsqueeze(0)))

        if self.detach_pathway_num is not None:
            embedding_spaces[self.detach_pathway_num] = embedding_spaces[self.detach_pathway_num].detach() # detaching features of q
        loss_contrastive = self.contrastive_loss(embedding_spaces)

        losses.update(loss_contrastive)
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



