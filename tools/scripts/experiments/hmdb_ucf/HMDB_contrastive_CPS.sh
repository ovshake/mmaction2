#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J hmdb_contrastive
#SBATCH -o ./out/hmdb_contrastive_CPS.out
#SBATCH -e ./out/hmdb_contrastive_CPS.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -w ai8
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------


# exp_name="tsm_k400_HMDB_two_pathway_CPS_C"
# exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_k400_100e_ucf_hmdb_color_p_speed_contrastive_color_p_speed_sgd_color_temp_5.py'

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/hmdb data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' total_epochs=100 --validate --deterministic

# exp_name="tsm_k400_HMDB_two_pathway_CPS_C"
# exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_k400_100e_ucf_hmdb_color_p_speed_contrastive_color_p_speed_sgd_color_temp_5.py'

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/hmdb data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' total_epochs=100 --validate --deterministic
#/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_HMDB_UCF_baseline/tsm_k400_HMDB_two_pathway_C_S/hmdb

exp_name1="tsm_k400_HMDB_two_pathway_CPS_C"
exp_section1="tsm_r50_1x1x3_100e_HMDB_UCF_baseline"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/hmdb_ucf/tsm_r50_1x1x3_100e_train_cls_batch_12.py'
exp_name="tsm_k400_HMDB_two_pathway_CPS_C"
exp_section="tsm_r50_1x1x3_100e_HMDB_UCF_baseline_cls"


PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/ data.train.domain='hmdb51' data.val.domain='hmdb51' data.test.domain='hmdb51' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/hmdb/latest.pth" total_epochs=100  --validate --deterministic
