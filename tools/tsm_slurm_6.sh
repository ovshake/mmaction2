#!/bin/bash 
#SBATCH -J kuh-multiple-contrastive-space
#SBATCH --gres=gpu:4
#SBATCH -o /data/abhishek/outputs/slurm_6.out 
#SBATCH -e /data/abhishek/outputs/slurm_6.err 
#SBATCH -t 13-0:00

. /data/abhishek/anaconda3/etc/profile.d/conda.sh 
conda activate action-dg 


exp_name="tsm-multiple-contrastive-space-k400-uh"
exp_section="tsm_r50_1x1x3_100e_k400_uh_rgb_v2"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_multiple_contrastive_space.py"


PORT=7073 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='kinetics' data.val.domain='kinetics' total_epochs=100 --validate


PORT=7073 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='ucf-hmdb' data.val.domain='ucf-hmdb' total_epochs=100 --validate

