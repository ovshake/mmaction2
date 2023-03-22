#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J check_loss
#SBATCH -o /data/jongmin/outputs/check_loss.out
#SBATCH -e /data/jongmin/outputs/check_loss.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[5,10]
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 64G


. /data/jongmin/anaconda3/etc/profile.d/conda.sh
conda activate action-dg





exp_name="overfit_07"
exp_section="overfit"
config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/overfit/tsm_r50_1x1x3_k400_100e_overfit.py"


bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='val' data.val.domain='val' total_epochs=500 --validate --deterministic
