#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J train_tsm_baseline_check
#SBATCH -o ./out/train_tsm_baseline_total_batch_92.out
#SBATCH -e ./out/train_tsm_baseline_total_batch_92.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -w ai12
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G



. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
exp_name="tsm_k400_100e_ekmmsada_dg_baseline_check"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_baseline"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_baseline_batch_12_check.py"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

#--------------------------------------------------
