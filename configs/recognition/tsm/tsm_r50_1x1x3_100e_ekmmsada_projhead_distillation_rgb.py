_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# frozen params

find_unused_parameters=True
# fp16 training
fp16 = dict()

# model settings
clip_len = 16
"""
cls_head=dict(
        type='TSMHead',
        num_classes=400,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True)

"""
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'
speed_model = dict(
            type='ColorSpatialSelfSupervised1SimSiamContrastiveHeadRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=None,
            contrastive_head=dict(type='ContrastiveHead',
                                num_segments=clip_len,
                                feature_size=2048),
            contrastive_loss=None)

color_model = dict(
            type='ColorSpatialSelfSupervised1SimSiamContrastiveHeadRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=None,
            contrastive_head=dict(type='ContrastiveHead',
                                num_segments=clip_len,
                                feature_size=2048),
            contrastive_loss=None)



model = dict(
            type='MultipleContrastiveDistillerRecognizer',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(num_segments=clip_len,
                        num_classes=8,
                        spatial_type=None,
                        in_channels=2048),
            contrastive_head=dict(type='ContrastiveHead',
                                num_segments=clip_len,
                                feature_size=2048),
            emb_loss=dict(type='EmbeddingLoss'),
            emb_stage='proj_layer',
            domain='D1',
            speed_network=speed_model,
            color_network=color_model
            )
# dataset settings
train_dataset = 'D1'
val_dataset = 'D1'
test_dataset = None
dataset_type = 'RawframeDataset'
dataset_type = 'EpicKitchensMMSADA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
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
        type=dataset_type,
        domain='D1',
        pipeline=train_pipeline,
        sample_by_class=True),
    val=dict(
        type=dataset_type,
        domain='D1',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        domain='D1',
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
work_dir = f'./work_dirs/test'
total_epochs = 100


log_config = dict(  # Config to register logger hook
    interval=5,  # Interval to print the log
    hooks=[  # Hooks to be implemented during training
        dict(type='TextLoggerHook'),  # The logger used to record the training process
        dict(type='TensorboardLoggerHook'),  # The Tensorboard logger is also supported
    ])