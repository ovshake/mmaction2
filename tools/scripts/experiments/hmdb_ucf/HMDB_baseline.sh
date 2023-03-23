#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J hmdb_baseline
#SBATCH -o ./out/hmdb_baseline.out
#SBATCH -e ./out/train_cosdsdsdsdncat_all.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -w ai8
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_two_pathway_C_S"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"
/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline/tsm_k400_HMDB_two_pathway_C_S/hmdb
PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline/tsm_k400_HMDB_two_pathway_C_S/hmdb/latest.pth" total_epochs=100  --validate --deterministic

# exp_name="tsm_k400_baseline_HMDB"
# exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_UCF_HMDB_baseline_rgb.py'

# bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ucf data.train.domain='hmdb101' data.val.domain='hmdb101' data.test.domain='hmdb101' total_epochs=100 --validate --deterministic


# exp_name="tsm_k400_HMDB_two_pathway_CPS_C"
# exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_k400_100e_ucf_hmdb_color_p_speed_contrastive_color_p_speed_sgd_color_temp_5.py'

# bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ucf data.train.domain='hmdb' data.val.domain='hmdb' total_epochs=100 --validate --deterministic


# exp_name="tsm_k400_HMDB_two_pathway_C_S"
# exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_k400_100e_ucf_hmdb_color_speed_contrastive_color_sgd_speed_temp_5.py'

# bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ucf data.train.domain='hmdb' data.val.domain='hmdb' total_epochs=100 --validate --deterministic



# exp_name1="tsm_k400_HMDB_two_pathway_CPS_C"
# exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
# exp_name="tsm_k400_HMDB_two_pathway_C_S"
# exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

# bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ucf data.train.domain='hmdb' data.val.domain='hmdb' total_epochs=100 --validate --deterministic


# xp_name1="tsm_k400_HMDB_two_pathway_C_S"
# exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
# exp_name="tsm_k400_HMDB_two_pathway_C_S"
# exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"

# bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ucf data.train.domain='hmdb' data.val.domain='hmdb' total_epochs=100 --validate --deterministic
