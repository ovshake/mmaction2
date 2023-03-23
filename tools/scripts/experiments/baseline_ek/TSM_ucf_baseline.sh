#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J ucf101
#SBATCH -o ./out/ucf.out
#SBATCH -e ./out/ucf.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -w ai13
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G



. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
exp_name="tsm_k400_100e_ekmmsada_dg_baseline_ucf"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_baseline"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_baseline_ucf.py"

python /data/shinpaul14/projects/mmaction2/tools/train.py $config --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 total_epochs=5 --validate --deterministic 

#PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='ucf' data.val.domain='ucf' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

#--------------------------------------------------
