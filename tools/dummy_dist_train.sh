#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J tsm-1
#SBATCH -o /data/abhishek/outputs/slurm_1.out
#SBATCH -e /data/abhishek/outputs/slurm_1.err
#SBATCH -t 15:00:00
#SBATCH -p batch
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 64G


# . /data/abhishek/anaconda3/etc/profile.d/conda.sh
# conda activate action-dg


exp_name="speed_plus_color_contrastive_xdb_no_proj_head"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_v1"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ekmmsada_multiple_contrastive_space.py"


PORT=7070 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 2 --cfg-options data.train.domain='D1' data.val.domain='D1' total_epochs=100 --validate

PORT=7070 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 2 --cfg-options data.train.domain='D2' data.val.domain='D2' total_epochs=100 --validate

PORT=7070 bash /data/abhishek/projects/mmaction2/tools/dist_train.sh $config 2 --cfg-options data.train.domain='D3' data.val.domain='D3' total_epochs=100 --validate