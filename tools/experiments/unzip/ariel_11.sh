#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -J unzip
#SBATCH -o ./out/unzip.out
#SBATCH -e ./out/unzip.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 10G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg

cd /local_datasets
unzip /local_datasets/EPIC_KITCHENS_UDA.zip