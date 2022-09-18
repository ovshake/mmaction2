# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseWeightedLoss
from .binary_logistic_regression_loss import BinaryLogisticRegressionLoss
from .bmn_loss import BMNLoss
from .cross_entropy_loss import BCELossWithLogits, CrossEntropyLoss
from .hvu_loss import HVULoss
from .nll_loss import NLLLoss
from .ohem_hinge_loss import OHEMHingeLoss
from .ssn_loss import SSNLoss
from .slowfast_selfsupervised_loss import SlowFastSelfSupervisedLoss, ContrastiveLoss, SingleInstanceContrastiveLoss, SingleInstanceContrastiveLossv2, SimSiamCosineSimLoss
from .multiple_contrastive_loss import MultipleContrastiveLoss, MultipleContrastiveSingleInstanceLoss, MultiplePathwayBaselineContrastiveLoss
from .moco_loss import MocoLoss


__all__ = [
    'BaseWeightedLoss', 'CrossEntropyLoss', 'NLLLoss', 'BCELossWithLogits',
    'BinaryLogisticRegressionLoss', 'BMNLoss', 'OHEMHingeLoss', 'SSNLoss',
    'HVULoss', 'SlowFastSelfSupervisedLoss', 'MultipleContrastiveLoss',
    'ContrastiveLoss', 'MocoLoss', 'SingleInstanceContrastiveLoss', 'MultipleContrastiveSingleInstanceLoss', 'SingleInstanceContrastiveLossv2', 'SimSiamCosineSimLoss', 'MultiplePathwayBaselineContrastiveLoss'
]
