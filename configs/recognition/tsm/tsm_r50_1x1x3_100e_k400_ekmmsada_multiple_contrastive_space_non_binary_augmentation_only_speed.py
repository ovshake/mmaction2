_base_ = [
    '../../_base_/models/tsm_r50.py', '../../_base_/schedules/sgd_tsm_50e.py',
    '../../_base_/default_runtime.py'
]

workflow = [('train', 1)]
# fp16 training 
fp16 = dict()

# model settings
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'
num_contrastive_heads = 2
model = dict(
            type='MultipleContrastiveRecognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True), # SyncBN
                shift_div=8),
            cls_head=dict(num_segments=1, num_classes=8, spatial_type=None, in_channels=128 * num_contrastive_heads), 
            num_contrastive_heads=num_contrastive_heads, 
            self_supervised_loss=dict(type='MultipleContrastiveLoss'), 
            contrastive_head=dict(type='TwoPathwayContrastiveHead',
                                feature_size=2048 * 7 * 7))

# dataset settings
dataset_type = 'RawframeDataset'
train_dataset_type = 'EpicKitchensMultiContrastiveNonBinaryAugSpaces'
val_dataset_type = 'EpicKitchensMMSADA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


contrastive_pipeline = [
    dict(type='RandomFrequencySampleFrames', clip_len=8, frame_interval=1, num_clips=1),
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
    videos_per_gpu=15,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=train_dataset_type,
        domain='D1',
        pipeline=contrastive_pipeline,
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
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])



# runtime settings
checkpoint_config = dict(interval=10)
work_dir = './work_dirs/tsm_r50_1x1x3_100e_k400_ucf_hmdb_rgb/slow-fast-contrastive-head/train_D1_test_D2/'
total_epochs = 500

# optimizer
optimizer = dict(
    lr=0.0075 * (15 / 8) * (4 / 8),  # this lr is used for 8 gpus
)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', step=[int(0.4 * total_epochs), int(0.8 * total_epochs)])
