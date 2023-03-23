#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J train_both_color_speed_contrastive_symmetric_cls
#SBATCH -o /data/shinpaul14/outputs/train_both_color_contrastive_symmetric.out
#SBATCH -e /data/shinpaul14/outputs/train_both_color_contrastive_symmetric.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -x ai[1,5,10]
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
# exp_name="tsm-k400-both-color-speed-contrastive"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_symmetric_5_V1"
# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/temp_5_symmetric/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_color_speed_contrastive_color_speed_temp_5.py'



# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic


exp_name1="tsm-k400-both-color-speed-contrastive"
exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
exp_name="tsm-k400-both-color-speed-contrastive"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_cls"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic

