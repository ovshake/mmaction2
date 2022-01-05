#!/bin/bash 
#SBATCH --job-name=mutlicontrastive-learning
#SBATCH --gres=gpu:4
#SBATCH -o /data/abhishek/outputs/slurm.out 
#SBATCH -e /data/abhishek/outputs/slurm.err 
#SBATCH -t 13-0:00
#SBATCH -w node1


. /data/abhishek/anaconda3/etc/profile.d/conda.sh 
conda activate action-dg 





exp_name="multicontrastive-learning-only-speed-fp16-scratch"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ekmmsada_multiple_contrastive_space_only_speed.py"


bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' --validate

# bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' --validate





