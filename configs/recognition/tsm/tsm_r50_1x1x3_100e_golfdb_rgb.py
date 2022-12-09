_base_ = [
    '../../_base_/schedules/sgd_tsm_50e.py', # '../../_base_/models/tsm_r50.py',
    '../../_base_/default_runtime.py'
]

# fp16 training
fp16 = dict()

# model settings
load_from = 'https://download.openmmlab.com/mmaction/recognition/tsm/tsm_r50_1x1x8_50e_kinetics400_rgb/tsm_r50_1x1x8_50e_kinetics400_rgb_20200607-af7fb746.pth'
# model = dict(
#             backbone=dict(type='ResNet',
#                 depth=50),
#             cls_head=dict(, num_segments=1, num_classes=9))


model = dict(
    type='Recognizer2D',
    backbone=dict(
        type='ResNet',
        pretrained='torchvision://resnet50',
        depth=50),
    cls_head=dict(
        type='TSMHead',
        num_classes=9,
        in_channels=2048,
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))


# dataset settings
dataset_type = 'GolfDB'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

data = dict(
    videos_per_gpu=48,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        data_file='/data/shinpaul14/projects/golfdb/data/train_split_1.pkl',
        vid_dir='/data/shinpaul14/data/videos_160',
        seq_length=1),
    val=dict(
        type=dataset_type,
        data_file='/data/shinpaul14/projects/golfdb/data/val_split_1.pkl',
        vid_dir='/data/shinpaul14/data/videos_160',
        seq_length=1,
        train=False),
    )

evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy', 'ece_score'])

# optimizer
optimizer = dict(
    lr=0.0075 * (48 / 8) * (4 / 8),  # this lr is used for 8 gpus
)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(policy='step', step=[40, 80])

# runtime settings
checkpoint_config = dict(interval=5)
work_dir = './work_dirs/test/'
total_epochs = 100
