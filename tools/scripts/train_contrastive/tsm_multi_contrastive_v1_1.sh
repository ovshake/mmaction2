#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J self_contrastive_xbd_keys_sgd
#SBATCH -o /data/shinpaul14/outputs/self_contrastive_xbd_keys_sgd.out
#SBATCH -e /data/shinpaul14/outputs/self_contrastive_xbd_keys_sgd.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -x ai[10]
#SBATCH --cpus-per-gpu 15
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg



# exp_name="tsm-k400-multi-contrastive-xbd-keys-self_same_start"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_slef_contrastive_v1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/self/tsm_r50_1x1x3_k400_100e_ekmmsada_rgb_contrastive_xbd_sgd_normal_frozen_cls_same_start.py"

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic



exp_name1="tsm-k400-multi-contrastive-xbd-keys-self_same_start"
exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_slef_contrastive_v1"

exp_name="tsm-k400-multi-contrastive-xbd-keys-self_same_start_cls_last"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_slef_contrastive_v2"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic

exp_name="tsm-k400-multi-contrastive-xbd-keys-self_same_start_cls_best"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_slef_contrastive_v2"
PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_80.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_75.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_100.pth" total_epochs=100  --validate --deterministic