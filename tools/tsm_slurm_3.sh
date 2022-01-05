#!/bin/bash 
#SBATCH --job-name=mutlicontrastive-learning
#SBATCH --gres=gpu:4
#SBATCH -o /data/abhishek/outputs/slurm_3.out 
#SBATCH -e /data/abhishek/outputs/slurm_3.err 
#SBATCH -t 13-0:00
#SBATCH -w node1


. /data/abhishek/anaconda3/etc/profile.d/conda.sh 
conda activate action-dg 


exp_name="multicontrastive-learning-fp16-all-way"
exp_section="tsm_r50_1x1x3_100e_k400_ucf_hmdb"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ucf_hmdb_multiple_contrastive_space.py"

PORT=7800 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_kinetics_test_kinetics data.train.domain='kinetics' data.val.domain='kinetics' --validate