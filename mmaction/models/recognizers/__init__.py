# Copyright (c) OpenMMLab. All rights reserved.
from .audio_recognizer import AudioRecognizer
from .base import BaseRecognizer
from .recognizer2d import (Recognizer2D, SlowFastSelfSupervisedRecognizer2D, VCOPSRecognizer2D,
                        SlowFastSelfSupervisedContrastiveHeadRecognizer2D,
                        ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D, MultipleContrastiveRecognizer2D)
from .recognizer3d import Recognizer3D
from .mocorecognizer2d import SlowFastSelfSupervisedMOCORecognizer2D, ColorSpatialSelfSupervisedMOCOContrastiveHeadRecognizer2D

from .augself_recognizer import ColorSpatialAugSelfSupervisedRecognizer2D, ColorSpatialAugSelfSupervisedContrastiveHeadRecognizer2D, MultipleContrastiveAugselfRecognizer2D

from .color_contrastive_recognizer2d import ColorSpatialSelfSupervised1ContrastiveHeadRecognizer2D, ColorSpatialSelfSupervised1SimSiamContrastiveHeadRecognizer2D, ColorSpatialSelfSupervised1SimSiamInversePredictorContrastiveHeadRecognizer2D

from .twopathway_recognizer2d import TwoPathwaySelfSupervised1SimSiamCosSimRecognizer2D

from .multiplepathway_recognizer2d import MultiplePathwaySelfSupervised1SimSiamCosSimRecognizer2D
from .multiple_contrastive_distiller import MultipleContrastiveDistillerRecognizer

__all__ = ['BaseRecognizer', 'Recognizer2D', 'Recognizer3D', 'AudioRecognizer', 'SlowFastSelfSupervisedRecognizer2D',
            'SlowFastSelfSupervisedContrastiveHeadRecognizer2D',
            'ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D',
            'MultipleContrastiveRecognizer2D', 'SlowFastSelfSupervisedMOCORecognizer2D',
            'ColorSpatialSelfSupervisedMOCOContrastiveHeadRecognizer2D',
            'ColorSpatialAugSelfSupervisedRecognizer2D', 'ColorSpatialAugSelfSupervisedContrastiveHeadRecognizer2D', 'MultipleContrastiveAugselfRecognizer2D', 'ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D',
            'ColorSpatialSelfSupervised1ContrastiveHeadRecognizer2D', 'ColorSpatialSelfSupervised1SimSiamContrastiveHeadRecognizer2D',
            'ColorSpatialSelfSupervised1SimSiamInversePredictorContrastiveHeadRecognizer2D',
            'TwoPathwaySelfSupervised1SimSiamCosSimRecognizer2D',
            'MultiplePathwaySelfSupervised1SimSiamCosSimRecognizer2D' , 'MultipleContrastiveDistillerRecognizer']
