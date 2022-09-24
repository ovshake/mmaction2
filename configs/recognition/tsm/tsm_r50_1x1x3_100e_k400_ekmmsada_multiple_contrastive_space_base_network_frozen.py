_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

workflow = [('train', 1)]
# fp16 training
fp16 = dict()
find_unused_parameters = True
# model settings
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'
model = dict(
            type='MultiplePathwaySelfSupervised1SimSiamCosSimRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                frozen_stages=4,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(num_segments=8,
                        num_classes=8,
                        spatial_type=None,
                        in_channels=2048),
            num_heads=4,
            contrastive_loss=dict(type='MultiplePathwayBaselineContrastiveLoss',
                                    use_row_sum_b=True,
                                    use_positives_in_denominator=True),
            contrastive_head=dict(type='ContrastiveHead',
                                num_segments=8,
                                feature_size=2048),
            detach_pathway_num=None,
            freeze_cls_head=True)

# dataset settings
dataset_type = 'RawframeDataset'
train_dataset_type = 'EpicKitchensMultipleContrastiveSpaces'
val_dataset_type = 'EpicKitchensMMSADA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

colorjitter_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='ColorJitter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
vanilla_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

fast_colorjitter_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='ColorJitter'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

fast_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=2, num_clips=1),
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
        clip_len=8,
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
        domain='D1',
        pipelines=[fast_colorjitter_pipeline,
        colorjitter_pipeline,
        fast_pipeline,
        vanilla_pipeline],
        test_mode=False,
        sample_by_class=True,
        ),
    val=dict(
        type=val_dataset_type,
        domain='D1',
        pipeline=val_pipeline),
    test=dict(
        type=val_dataset_type,
        domain='D1',
        pipeline=val_pipeline,
        filename_tmpl='frame_{:010d}.jpg',
    ))

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy', 'ece_score'])

# optimizer

optimizer = dict(
    type='SGD',
    constructor='TSMOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=0.0075 * (12 / 8) * (4 / 8),  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/test'
total_epochs = 100
