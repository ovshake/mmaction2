#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J train_late_fusion_color_speed_d3
#SBATCH -o /data/jongmin/outputs/train_late_fusion_d3.out
#SBATCH -e /data/jongmin/outputs/train_late_fusion_d3.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[5,10]
#SBATCH --cpus-per-gpu 16
#SBATCH --mem 64G


. /data/jongmin/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


#--------------------------------------------------
# exp_name="tsm-k400-color-contrastive_xd_sgd_color_speed_vcop_fusion"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_speed_late_fusion"
# config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/late_fusion/tsm_r50_1x1x3_100e_latefusion_speed_color_vcop_input_normal_cls.py'

exp_name="tsm-k400-color-contrastive_xd_sgd_fusion_norm_color_speed_concat"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_speed_late_fusion"
config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/late_fusion/tsm_r50_1x1x3_100e_latefusion_speed_color_concat.py'
#PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic


# exp_name1="tsm-k400-color-contrastive_xd_sgd_color_temp_5_first"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
# exp_name="tsm-k400-color-contrastive_xd_sgd_color_temp_5_first"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_cls"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic
