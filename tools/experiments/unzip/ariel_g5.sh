#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -J unzip
#SBATCH -o /data/shinpaul14/projects/mmaction2/unzip.out
#SBATCH -e /data/shinpaul14/projects/mmaction2/unzip.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -w ariel-g5
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 20G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg

cd /local_datasets
unzip /local_datasets/EPIC_KITCHENS_UDA.zip
