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


#--------------------- CLIP

# using clip feature to train the cls head. only training cls head. 
@RECOGNIZERS.register_module()
class CLIP_Recognizer2D(Recognizer2D):
    def __init__(self,
                 backbone,
                 cls_head=None,
                 neck=None,
                 train_cfg=None,
                 clip_method = None,
                 test_cfg=None):
        super().__init__(backbone=backbone, cls_head=cls_head, train_cfg=train_cfg, test_cfg=test_cfg)
        self.clip_method = clip_method
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("RN50", device='cpu')
        self.clip_model.visual.attnpool.c_proj = torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
        self.clip_model.to(device)
        #self.clip_model.half()
        self.clip_model.float()



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
        #self.color_to_vanilla_projection_layer = nn.Linear(self.contrastive_head.img_dim,
                                   #             self.contrastive_head.img_dim, bias=True)

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
        # imgs_pathway_A = imgs_pathway_A.reshape((-1, ) + imgs_pathway_A.shape[2:])
        num_segs = imgs.shape[0] // batches
        # print('imgs.shape', imgs.shape)
        device = "cuda" if torch.cuda.is_available() else "cpu"
   
        with torch.no_grad():
            for i in range(imgs.shape[0]):
                for j in range(imgs.shape[1]):

                    if self.clip_method=='all':
                        single_frame = imgs[i][j]
                        if i == 0 and j == 0:
                            clip_image = single_frame.unsqueeze(0).to(device)
                            print('clip_image size  - ', clip_image.shape)
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
                        if i == 0 and j == int(imgs.shape[1]/2):# in 8 frame setting, 4th frame is the center frame
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

        # print('clip_features - ', clip_features.shape)
        # print(self.cls_head)
        losses = dict()
        clip_features = clip_features.to(device).float()
        #check_ = torch.rand(96,2048).to(device).float()
        print('clip_features', clip_features.shape)
        print('num_segs', num_segs)
        cls_score_pathway_A = self.cls_head(clip_features.detach(), num_segs)
        # print('cls_score_pathway_A',cls_score_pathway_A)
        gt_labels = labels.squeeze()
        # print('passed this parts ----------')

        loss_cls = self.cls_head.loss(cls_score_pathway_A, gt_labels, **kwargs)
      


        #print(loss_cls)
        #if we update the cls loss the model falls in to the wrong place
        losses.update(loss_cls) 
   
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
        # imgs_slow, imgs_fast = data_batch[0]['imgs'], data_batch[1]['imgs']
        imgs_slow= data_batch['imgs']
        label = data_batch['label']

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self((imgs_slow), label, return_loss=True)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches
        # print('imgs shape ~~~~~~~~- ', imgs.shape)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        for i in range(imgs.shape[0]):
            single_frame = imgs[i]
            if i == 0:
                clip_image = single_frame.unsqueeze(0).to(device)
                clip_features=self.clip_model.encode_image(clip_image).detach()
            else:
                clip_image = single_frame.unsqueeze(0).to(device)
                second_clip_features = self.clip_model.encode_image(clip_image).detach()
                clip_features = torch.cat((clip_features, second_clip_features), 0).detach()

        # print('clip_features in test time - ', clip_features.shape)
        clip_features = clip_features.to(device).float()
        cls_score = self.cls_head(clip_features, num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score