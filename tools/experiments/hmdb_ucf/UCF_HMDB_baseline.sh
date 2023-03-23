#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J HMDB_UCF_baseline
#SBATCH -o ./out/HMDB_UCF_baseline.out
#SBATCH -e ./out/HMDB_UCF_baseline.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -w ai8
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
exp_name="tsm_k400_baseline_HMDB"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_UCF_HMDB_baseline_rgb.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ucf data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' total_epochs=100 --validate --deterministic

exp_name="tsm_k400_baseline_UCF"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_UCF_HMDB_baseline_rgb.py'

bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ucf data.train.domain='ucf101' data.val.domain='ucf101' data.test.domain='ucf101' total_epochs=100 --validate --deterministic

