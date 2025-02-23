_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

# fp16 training 
fp16 = dict()

# model settings
clip_len = 16


# model settings
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'
model = dict(
            type='ColorSpatialSelfSupervisedContrastiveHeadRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(num_segments=clip_len, 
                        num_classes=38, 
                        spatial_type=None, 
                        in_channels=2048), 
            vanilla_contrastive_head=dict(type='ContrastiveHead',
                                num_segments=clip_len,
                                feature_size=2048), 
            color_contrastive_head=dict(type='ContrastiveHead',
                                num_segments=clip_len,
                                feature_size=2048), 
            contrastive_loss=dict(type='ContrastiveLoss', 
                                name='color'))


# dataset settings
train_dataset = 'kinetics'
val_dataset = 'kinetics'
clip_len = 16
test_dataset = None
dataset_type = 'RawframeDataset'
train_dataset_type = 'Kinetics400UCFHMDBTwoPathway'
val_dataset_type = 'Kinetics400UCFHMDB'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
pathway_A_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

pathway_B_pipeline = [
    dict(type='SampleFrames', clip_len=clip_len, frame_interval=1, num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='ColorJitter'),
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
    videos_per_gpu=10,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=train_dataset_type,
        domain=train_dataset,
        sample_by_class=True,
        pathway_A_pipeline=pathway_A_pipeline, 
        pathway_B_pipeline=pathway_B_pipeline),
    val=dict(
        type=val_dataset_type,
        domain=val_dataset,
        pipeline=val_pipeline), 
    test=dict(
        type=val_dataset_type,
        domain=test_dataset if test_dataset else val_dataset,
        pipeline=val_pipeline, 
        filename_tmpl='img_{:05d}.jpg',
    ))

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(
    lr=0.0075 * (4 / 8) * (10 / 8),  # this lr is used for 8 gpus
)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])

# runtime settings
checkpoint_config = dict(interval=5)
if test_dataset:
    work_dir = f'./work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/colorspatialselfsupervised/train_{train_dataset}_test_{test_dataset}/'
else:
    work_dir = f'./work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb/colorspatialselfsupervised/train_{train_dataset}_test_{val_dataset}/'
total_epochs = 100
