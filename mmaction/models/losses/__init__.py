# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseWeightedLoss
from .binary_logistic_regression_loss import BinaryLogisticRegressionLoss
from .bmn_loss import BMNLoss
from .cross_entropy_loss import BCELossWithLogits, CrossEntropyLoss
from .hvu_loss import HVULoss
from .nll_loss import NLLLoss
from .ohem_hinge_loss import OHEMHingeLoss
from .ssn_loss import SSNLoss
from .slowfast_selfsupervised_loss import SlowFastSelfSupervisedLoss, ContrastiveLoss, SingleInstanceContrastiveLoss, SingleInstanceContrastiveLossv2, SimSiamCosineSimLoss, SingleInstanceContrastiveLossv2_moco_t, SimSiamLoss
from .multiple_contrastive_loss import MultipleContrastiveLoss, MultipleContrastiveSingleInstanceLoss, MultiplePathwayBaselineContrastiveLoss, MultiplePathwayBaselineContrastiveLoss_div ,Multi_Contrastive_Loss_each_space#Multi_Contrastive_Loss
from .multi_contrastive_loss import multi_contrastive_loss #Multi_Contrastive_Loss
from .simsiam_loss import Symmetric_ContrastiveLossv2, Asymmetric_ContrastiveLossv2, Symmetric_ContrastiveLossv2_div
from .moco_loss import MocoLoss
from .embedding_loss import EmbeddingLoss
from .supcon_loss import SupConLoss
from .sup_con_loss import symmetric_supervisedContrastiveLoss, supervisedContrastiveLoss


__all__ = [
    'BaseWeightedLoss', 'CrossEntropyLoss', 'NLLLoss', 'BCELossWithLogits','SupConLoss', 'EmbeddingLoss',
    'BinaryLogisticRegressionLoss', 'BMNLoss', 'OHEMHingeLoss', 'SSNLoss',
    'HVULoss', 'SlowFastSelfSupervisedLoss', 'MultipleContrastiveLoss', 'SingleInstanceContrastiveLossv2_moco_t'
    'ContrastiveLoss', 'MocoLoss', 'SingleInstanceContrastiveLoss', 'MultipleContrastiveSingleInstanceLoss', 'SingleInstanceContrastiveLossv2', 'SimSiamCosineSimLoss', 'MultiplePathwayBaselineContrastiveLoss'
    'SimSiamLoss','multi_contrastive_loss','MultiplePathwayBaselineContrastiveLoss_div', 'Multi_Contrastive_Loss_each_space','Symmetric_ContrastiveLossv2',
    'Asymmetric_ContrastiveLossv2','Symmetric_ContrastiveLossv2_div'
    'symmetric_supervisedContrastiveLoss', 'supervisedContrastiveLoss',
]
