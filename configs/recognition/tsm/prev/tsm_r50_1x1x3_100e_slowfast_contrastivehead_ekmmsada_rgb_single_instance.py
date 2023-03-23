_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# fp16 training 
fp16 = dict()

# model settings
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'
fast_clip_len = 16 
slow_clip_len = 8
model = dict(
            type='SlowFastSelfSupervisedContrastiveHeadRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(num_segments=16, 
                        num_classes=8, 
                        in_channels=2048, 
                        spatial_type=None), 
            contrastive_loss=dict(type='SingleInstanceContrastiveLoss', 
                            name='slowfast'),
            slow_contrastive_head=dict(type='ContrastiveHead',
                                num_segments=slow_clip_len,
                                feature_size=2048
                                ), 
            fast_contrastive_head=dict(type='ContrastiveHead', 
                                num_segments=fast_clip_len, 
                                feature_size=2048
                                ))

# dataset settings
dataset_type = 'RawframeDataset'
train_dataset_type = 'EpicKitchensSlowFastMMSADA'
val_dataset_type = 'EpicKitchensMMSADA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
slow_train_pipeline = [
    dict(type='SampleFrames', clip_len=slow_clip_len, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

fast_train_pipeline = [
    dict(type='SampleFrames', clip_len=fast_clip_len, frame_interval=1, num_clips=1),
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
        clip_len=fast_clip_len,
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
    videos_per_gpu=10,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=10),
    train=dict(
        type=train_dataset_type,
        domain='D1',
        sample_by_class=True,
        slow_pipeline=slow_train_pipeline, 
        fast_pipeline=fast_train_pipeline),
    val=dict(
        type=val_dataset_type,
        domain='D1',
        pipeline=val_pipeline), 
    test=dict(
        type=val_dataset_type,
        domain='D2',
        pipeline=val_pipeline
    ))

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy', 'ece_score'])

# optimizer
num_gpus = 4
optimizer = dict(
    lr=0.0075 * (10 / 8) * (num_gpus / 8),  # this lr is used for 8 gpus
)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/test'
total_epochs = 100
