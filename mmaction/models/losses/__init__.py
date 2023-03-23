# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseWeightedLoss
from .binary_logistic_regression_loss import BinaryLogisticRegressionLoss
from .bmn_loss import BMNLoss
from .cross_entropy_loss import (BCELossWithLogits, CBFocalLoss,
                                 CrossEntropyLoss)
from .hvu_loss import HVULoss
from .nll_loss import NLLLoss
from .ohem_hinge_loss import OHEMHingeLoss
from .ssn_loss import SSNLoss
from .slowfast_selfsupervised_loss import SlowFastSelfSupervisedLoss, ContrastiveLoss, SingleInstanceContrastiveLoss, SingleInstanceContrastiveLossv2, SimSiamCosineSimLoss, SingleInstanceContrastiveLossv2_moco_t, SimSiamLoss, SingleInstanceContrastiveLossv2_sum_positive
from .multiple_contrastive_loss import MultipleContrastiveLoss, MultipleContrastiveSingleInstanceLoss, MultiplePathwayBaselineContrastiveLoss, MultiplePathwayBaselineContrastiveLoss_div ,Multi_Contrastive_Loss_each_space#Multi_Contrastive_Loss
from .multi_contrastive_loss import multi_contrastive_loss #Multi_Contrastive_Loss
from .simsiam_loss import Symmetric_ContrastiveLossv2, Asymmetric_ContrastiveLossv2, Symmetric_ContrastiveLossv2_div
from .moco_loss import MocoLoss
from .embedding_loss import EmbeddingLoss
from .supcon_loss import SupConLoss 
from .contrastive_loss_add_positive import SingleInstanceContrastiveLossv2_add_positive
from .sup_con_loss import supervisedContrastiveLoss,symmetric_supervisedContrastiveLoss

__all__ = [
    'BaseWeightedLoss', 'CrossEntropyLoss', 'NLLLoss', 'BCELossWithLogits', 'EmbeddingLoss','SingleInstanceContrastiveLossv2_add_positive',
    'BinaryLogisticRegressionLoss', 'BMNLoss', 'OHEMHingeLoss', 'SSNLoss','SupConLoss',
    'HVULoss', 'SlowFastSelfSupervisedLoss', 'MultipleContrastiveLoss', 'SingleInstanceContrastiveLossv2_moco_t'
    'ContrastiveLoss', 'MocoLoss', 'SingleInstanceContrastiveLoss', 'MultipleContrastiveSingleInstanceLoss', 'SingleInstanceContrastiveLossv2', 'SimSiamCosineSimLoss', 'MultiplePathwayBaselineContrastiveLoss'
    'SimSiamLoss','multi_contrastive_loss','MultiplePathwayBaselineContrastiveLoss_div', 'Multi_Contrastive_Loss_each_space','Symmetric_ContrastiveLossv2',
    'Asymmetric_ContrastiveLossv2','Symmetric_ContrastiveLossv2_div','SingleInstanceContrastiveLossv2_sum_positive'
    ,'Supervised_contrastive_loss','supervisedContrastiveLoss', 'symmetric_supervisedContrastiveLoss',
]
