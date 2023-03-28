#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J baseline
#SBATCH -o /data/shinpaul14/outputs/baseline.out
#SBATCH -e /data/shinpaul14/outputs/baseline.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -w ariel-v13
#SBATCH --cpus-per-gpu 8
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate dg



exp_name="tsm_k400_baseline"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_reproduce"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_rgb.py"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic
