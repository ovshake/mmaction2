_base_ = [ '../../../_base_/models/tsm_r50.py', '../../../_base_/schedules/sgd_tsm_50e.py', '../../../_base_/default_runtime.py' ]

find_unused_parameters=True
# fp16 training
fp16 = dict()

# model settings
clip_len = 8

CPS_model = dict(
            type='SimSiamRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(type='TSMHead',
                        num_segments=clip_len,
                        num_classes=8,
                        spatial_type=None,
                        consensus=dict(type='AvgConsensus', dim=1),
                        in_channels=2048,
                        init_std=0.001,
                        dropout_ratio=0.0),
            projectionMLP=dict(type='projection_MLP',
                                num_segments=clip_len,
                                feature_size=2048),
            predictionMLP = dict(type='prediction_MLP',
                                feature_size=2048),
            contrastive_loss=dict(type='SingleInstanceContrastiveLossv2',
                                name='color',temperature=5.0,
                                use_positives_in_denominator=True,
                              ))


color_model = dict(
            type='SimSiamRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(type='TSMHead',
                        num_segments=clip_len,
                        num_classes=8,
                        spatial_type=None,
                        consensus=dict(type='AvgConsensus', dim=1),
                        in_channels=2048,
                        init_std=0.001,
                        dropout_ratio=0.0),
            projectionMLP=dict(type='projection_MLP',
                                num_segments=clip_len,
                                feature_size=2048),
            predictionMLP = dict(type='prediction_MLP',
                                feature_size=2048),
            contrastive_loss=dict(type='SingleInstanceContrastiveLossv2',
                                name='color',temperature=5.0,
                                use_positives_in_denominator=True,
                              ))
vcops_num_clips=4
vcop_model = dict(
                type='VCOPSRecognizer2D_cls_no',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,frozen_stages=4,
                norm_cfg=dict(type='SyncBN', requires_grad=True), # not sure about this 
                shift_div=8),
            num_clips=3,
            cls_head=None,
            vcop_head=None)

model = dict(
            type='LateFusionRecognizer_all_in_one',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,frozen_stages=4,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=None,
            CPS_network=None,
            color_network=color_model,
            normal_network=CPS_model,
            vcop_network=vcop_model,
            domain='D1',
            )

# dataset settings
train_dataset = 'D1'
val_dataset = 'D1'
test_dataset = None
dataset_type = 'RawframeDataset'
train_dataset_type = 'EpicKitchensTemporalSpatialMMSADA_ensemble'
val_dataset_type = 'EpicKitchensMMSADA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


#color_jitter_pipeline = [
input_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

# fast_train_pipeline = [
#     dict(type='SampleFrames', clip_len=clip_len, frame_interval=2, num_clips=1),
#     dict(type='RawFrameDecode'),
#     dict(type='Resize', scale=(-1, 256)),
#     dict(type='RandomCrop', size=224),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='FormatShape', input_format='NCHW'),
#     dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
#     dict(type='ToTensor', keys=['imgs', 'label'])
# ]

val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=clip_len,
        frame_interval=1,
        num_clips=5,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=12,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=train_dataset_type,
        domain=train_dataset,
        pathway_A=input_pipeline,
        # pathway_B=input_pipeline,
        # pathway_C=input_pipeline,
        clip_len=clip_len),
    val=dict(
        type=val_dataset_type,
        domain=val_dataset,
        pipeline=val_pipeline),
    test=dict(
        type=val_dataset_type,
        domain=test_dataset if test_dataset else val_dataset,
        pipeline=val_pipeline
    ))

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy', 'ece_score'])

# optimizer
optimizer = dict(
    lr=0.0075 * (4 / 8) * (12 / 8),  # this lr is used for 8 gpus
)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/test'

