# Copyright (c) OpenMMLab. All rights reserved.
from .audio_recognizer import AudioRecognizer
from .base import BaseRecognizer
from .recognizer2d import (Recognizer2D, SlowFastSelfSupervisedRecognizer2D, VCOPSRecognizer2D, VCOPSRecognizer2D_cls_no,
                        SlowFastSelfSupervisedContrastiveHeadRecognizer2D,
                        ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D, MultipleContrastiveRecognizer2D)
from .recognizer3d import Recognizer3D
from .mocorecognizer2d import SlowFastSelfSupervisedMOCORecognizer2D, ColorSpatialSelfSupervisedMOCOContrastiveHeadRecognizer2D

from .augself_recognizer import ColorSpatialAugSelfSupervisedRecognizer2D, ColorSpatialAugSelfSupervisedContrastiveHeadRecognizer2D, MultipleContrastiveAugselfRecognizer2D

from .color_contrastive_recognizer2d import ColorSpatialSelfSupervised1ContrastiveHeadRecognizer2D, ColorSpatialSelfSupervised1SimSiamContrastiveHeadRecognizer2D, ColorSpatialSelfSupervised1SimSiamInversePredictorContrastiveHeadRecognizer2D, SimSiamRecognizer2D, SimSiamRecognizerWithSimSiamLoss2D

from .CLIP_recognizer2d import CLIP_Recognizer2D

from .twopathway_recognizer2d import TwoPathwaySelfSupervised1SimSiamCosSimRecognizer2D

from .multiplepathway_recognizer2d import MultiplePathwaySelfSupervised1SimSiamCosSimRecognizer2D
from .multiple_contrastive_distiller import MultipleContrastiveDistillerRecognizer, MultipleContrastiveDistillerRecognizer_4
from .multiple_contrastive_distiller_clip import MultipleContrastiveDistillerRecognizer_w_clip, MultipleContrastiveDistillerRecognizer_clip_ucf_hmdb,MultipleContrastiveDistillerRecognizer_w_clip_speed
from .recognizer2d_no_cls import Recognizer2D_no_cls , VCOPSRecognizer2D_no_cls, SimSiamRecognizer2D_no_cls, SimSiamRecognizer2D_vinilla
from .multi_simsiam_recognizer2d import Mult_SimSiam_Recognizer2D
from .late_fusion2d import LateFusionRecognizer, LateFusionRecognizer_all,LateFusionRecognizer_vcop, LateFusionRecognizer_norm_before, LateFusionRecognizer_norm_after, LateFusionRecognizer_combine_all, LateFusionRecognizer_combine_two,LateFusionRecognizer_combine_speed_color
from .late_fusion2d_v2_clip import LateFusionRecognizer_all_in_one_clip, LateFusionRecognizer_all_in_one_ucf_hmdb_clip

from .multiple_response_base_distiller import Multiple_response_baseDistillerRecognizer

from .check_shape import check_shape_Recognizer2D
from .L2D import L2DRecognizer2D
from .late_fusion2d_v2 import LateFusionRecognizer_all_in_one, LateFusionRecognizer_all_in_one_ucf_hmdb
from .simsiam_contrastive_recognizer2d import SimSiamRecognizer2D_add_positive_2_denominator
from .multiple_distiller import Multiple_Distiller_Recognizer
from .teacher_ensemble import Teacher_ensemble
from .asymmetric_simsiam_recognizer2d import asymmetrical_SimSiamRecognizer2D
__all__ = ['BaseRecognizer', 'Recognizer2D', 'Recognizer3D', 'AudioRecognizer', 'SlowFastSelfSupervisedRecognizer2D','LateFusionRecognizer_all_in_one_ucf_hmdb','CLIP_Recognizer2D',
            'SlowFastSelfSupervisedContrastiveHeadRecognizer2D','L2DRecognizer2D','MultipleContrastiveDistillerRecognizer_4',
            'ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D','LateFusionRecognizer_all_in_one','MultipleContrastiveDistillerRecognizer_w_clip_speed',
            'MultipleContrastiveRecognizer2D', 'SlowFastSelfSupervisedMOCORecognizer2D', 'MultipleContrastiveDistillerRecognizer_clip_ucf_hmdb', 'MultipleContrastiveDistillerRecognizer_w_clip',
            'ColorSpatialSelfSupervisedMOCOContrastiveHeadRecognizer2D', 'check_shape_Recognizer2D', 'SimSiamRecognizer2D_add_positive_2_denominator'
            'ColorSpatialAugSelfSupervisedRecognizer2D', 'ColorSpatialAugSelfSupervisedContrastiveHeadRecognizer2D', 'MultipleContrastiveAugselfRecognizer2D', 'ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D',
            'ColorSpatialSelfSupervised1ContrastiveHeadRecognizer2D', 'ColorSpatialSelfSupervised1SimSiamContrastiveHeadRecognizer2D',
            'ColorSpatialSelfSupervised1SimSiamInversePredictorContrastiveHeadRecognizer2D','LateFusionRecognizer_all_in_one_clip', 
            'TwoPathwaySelfSupervised1SimSiamCosSimRecognizer2D','SimSiamRecognizer2D_no_cls','LateFusionRecognizer_all_in_one_ucf_hmdb_clip',
            'MultiplePathwaySelfSupervised1SimSiamCosSimRecognizer2D' , 'MultipleContrastiveDistillerRecognizer','SimSiamRecognizer2D',
            'VCOPSRecognizer2D_no_cls', 'VCOPSRecognizer2D', 'Recognizer2D_no_cls', 'SimSiamRecognizer2D_vinilla','Mult_SimSiam_Recognizer2D',
            'SimSiamRecognizerWithSimSiamLoss2D', 'LateFusionRecognizer','LateFusionRecognizer_all','LateFusionRecognizer_vcop', 'LateFusionRecognizer_norm_before', 'LateFusionRecognizer_norm_after','LateFusionRecognizer_combine_all'
            ,'LateFusionRecognizer_combine_two','LateFusionRecognizer_combine_speed_color',
            'asymmetrical_SimSiamRecognizer2D',
            'Multiple_response_baseDistillerRecognizer','Multiple_Distiller_Recognizer',
            'Teacher_ensemble',             
            ]
