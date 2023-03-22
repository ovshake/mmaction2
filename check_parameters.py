import torch
from torch import nn

from mmaction.models import builder
from mmaction.models.builder import RECOGNIZERS
from mmaction.models.recognizers.recognizer2d import Recognizer2D
from mmaction.models.recognizers.base import BaseRecognizer

model = dict(
            type='SimSiamRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(num_segments=8,
                        num_classes=8,
                        spatial_type=None,
                        dropout_ratio=0.0,
                        in_channels=2048),
            projectionMLP=dict(type='projection_MLP',
                                num_segments=8,
                                feature_size=2048),
            predictionMLP = dict(type='prediction_MLP',
                                feature_size=2048),
            contrastive_loss=dict(type='SingleInstanceContrastiveLossv2',
                                name='color',
                                use_positives_in_denominator=True,
                                use_row_sum_b=True))
check_point = '/data/jongmin/projects/mmaction2_paul_work/work_dirs/trash1/trash1/train_D1_test_D1/epoch_5.pth'
model = builder.build_model(model)
model.load_state_dict(torch.load(check_point)['state_dict'])
print(model)
