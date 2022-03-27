#!/bin/bash 
#SBATCH -J color-jitter-D3
#SBATCH --gres=gpu:4
#SBATCH -o /data/abhishek/outputs/slurm_4.out 
#SBATCH -e /data/abhishek/outputs/slurm_4.err 
#SBATCH -t 13-0:00

. /data/abhishek/anaconda3/etc/profile.d/conda.sh 
conda activate action-dg 


exp_name="color-contrastive-head-all-gather"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_rgb.py"

PORT=7071 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100 --validate


PORT=7071 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100 --validate