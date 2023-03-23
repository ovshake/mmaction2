#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J check_loss
#SBATCH -o /data/shinpaul14/outputs/check_loss.out
#SBATCH -e /data/shinpaul14/outputs/check_loss.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[1,5,10]
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg





exp_name="overfit_07"
exp_section="overfit"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/overfit/tsm_r50_1x1x3_k400_100e_overfit.py"


bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='val' data.val.domain='val' total_epochs=500 --validate --deterministic
