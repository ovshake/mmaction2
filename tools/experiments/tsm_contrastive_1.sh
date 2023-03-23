#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J train_speed_contrastive
#SBATCH -o /data/shinpaul14/outputs/train_speed_contrastive.out
#SBATCH -e /data/shinpaul14/outputs/train_speed_contrastive.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[1,5,10]
#SBATCH --cpus-per-gpu 15
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
exp_name="tsm-k400-speed-contrastive_xd_sgd_speed"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_speed_frozen_cls.py"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=1 temp=2.0  --validate --deterministic
