#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J contrastive_xd_sgd_color
#SBATCH -o /data/shinpaul14/outputs/contrastive_xd_sgd_color.out
#SBATCH -e /data/shinpaul14/outputs/contrastive_xd_sgd_color.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[1,5,10]
#SBATCH --cpus-per-gpu 15
#SBATCH --mem 64G


. /data/shinpaul14/anaconda3/etc/profile.d/conda.sh
conda activate action-dg



# exp_name1="tsm-k400-color-contrastive_x_sgd_color"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_x_sgd_color"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_100.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_45.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name1="tsm-k400-color-contrastive_x_sgd_normal"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_x_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"
# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_100.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_45.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name1="tsm-k400-color-contrastive_xabd_sgd_color"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_xabd_sgd_color"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"
# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name1="tsm-k400-color-contrastive_xabd_sgd_normal"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_xabd_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"
# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_75.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_45.pth" total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name1="tsm-k400-color-contrastive_xad_sgd_color"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_xad_sgd_color"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"
# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_70.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_100.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_100.pth" total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name1="tsm-k400-color-contrastive_xad_sgd_normal"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_xad_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"
# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_100.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_60.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_5.pth" total_epochs=100  --validate --deterministic

# #--------------------------------------------------
# exp_nam1e="tsm-k400-color-contrastive_xbd_sgd_color"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_xbd_sgd_color"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"
# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_95.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_85.pth" total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name1="tsm-k400-color-contrastive_xbd_sgd_normal"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_xbd_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"
# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_50.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_100.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_60.pth" total_epochs=100  --validate --deterministic

#--------------------------------------------------

exp_name1="tsm-k400-color-contrastive_xd_sgd_color"
exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
exp_name="tsm-k400-color-contrastive_xd_sgd_color"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_60.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_65.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_80.pth" total_epochs=100  --validate --deterministic

#--------------------------------------------------
exp_name="tsm-k400-color-contrastive_xd_sgd_color_last_epoch"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic


#--------------------------------------------------

exp_name1="tsm-k400-color-contrastive_xd_sgd_color"
exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_rgb_scratch.py"
exp_name="tsm-k400-color-contrastive_xd_sgd_color"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_end"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_60.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_65.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_80.pth" total_epochs=100  --validate --deterministic

#--------------------------------------------------
exp_name="tsm-k400-color-contrastive_xd_sgd_color_last_epoch"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_end"

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic
#--------------------------------------------------


