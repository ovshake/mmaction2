#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J EK_CLIP_D1
#SBATCH -o ./out/D1.out
#SBATCH -e ./out/D1.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -w ariel-v11
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
exp_name="tsm_CLIP_all_frames_all"
exp_section="CLIP_teacher"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/CLIP/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' model.clip_method='all' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' model.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' model.domain='D3' total_epochs=100 --validate --deterministic 
