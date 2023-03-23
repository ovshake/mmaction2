_base_ = [ '../../../_base_/models/tsm_r50.py', '../../../_base_/schedules/sgd_tsm_50e.py', '../../../_base_/default_runtime.py' ]


# fp16 training
fp16 = dict()

# model settings
clip_len = 8

# model settings
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'
model = dict(
            type='Mult_SimSiam_Recognizer2D',
            backbone=dict(type='ResNetTSM',
                depth=50,
                norm_eval=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                shift_div=8),
            cls_head=dict(num_segments=clip_len,
                        num_classes=8,
                        spatial_type=None,
                        in_channels=2048,
                        dropout_ratio=0.0),
            num_heads=4,
            contrastive_loss=dict(type='Multi_Contrastive_Loss_each_space',
                                    use_row_sum_a=True,
                                    use_row_sum_b=True,
                                    use_positives_in_denominator=True),
            projectionMLP=dict(type='projection_MLP_multi',
                                num_segments=clip_len,
                                feature_size=2048),
            predictionMLP = dict(type='prediction_MLP_multi',
                                feature_size=2048),
            freeze_cls_head=True,
            detach_pathway_num=[1,2,3])

# dataset settings
dataset_type = 'RawframeDataset'
train_dataset_type = 'EpicKitchensMultipleContrastiveSpaces'
val_dataset_type = 'EpicKitchensMMSADA'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

query_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1, temporal_aug=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='ColorJitter_video'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
key0_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1,multi_path_aug=True,),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='ColorJitter_video', multi_color_aug_k=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

key1_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1, temporal_aug=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='ColorJitter_video', multi_color_aug_k1=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

key2_pipeline = [
    dict(type='SampleFrames', clip_len=8, frame_interval=1, num_clips=1, multi_path_aug_k1=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandomCrop', size=224),
    dict(type='ColorJitter_video'),
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
    videos_per_gpu=10,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=train_dataset_type,
        domain='D1',
        pipelines=[query_pipeline,key0_pipeline, key1_pipeline,key2_pipeline],
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
    constructor='TSMFreezeFCLayerOptimizerConstructor',
    paramwise_cfg=dict(fc_lr5=False),
    lr=0.0075 * (12 / 8) * (4 / 8),  # this lr is used for 8 gpus
    momentum=0.9,
    weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/test'
total_epochs = 100
