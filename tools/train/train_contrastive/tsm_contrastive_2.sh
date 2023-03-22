#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J train_speed_contrastive
#SBATCH -o /data/jongmin/outputs/train_speed_contrastive.out
#SBATCH -e /data/jongmin/outputs/train_speed_contrastive.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[5,10]
#SBATCH --cpus-per-gpu 15
#SBATCH --mem 64G


. /data/jongmin/anaconda3/etc/profile.d/conda.sh
conda activate action-dg



# exp_name="tsm-k400-speed-contrastive_x_sgd_speed"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_x_sgd_speed_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name="tsm-k400-speed-contrastive_x_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_x_sgd_normal_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name="tsm-k400-speed-contrastive_xabd_sgd_speed"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xabd_sgd_speed_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name="tsm-k400-speed-contrastive_xabd_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xabd_sgd_normal_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name="tsm-k400-speed-contrastive_xad_sgd_speed"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xad_sgd_speed_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name="tsm-k400-speed-contrastive_xad_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xad_sgd_normal_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

# #--------------------------------------------------
# exp_name="tsm-k400-speed-contrastive_xbd_sgd_speed"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xbd_sgd_speed_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

# #--------------------------------------------------

# exp_name="tsm-k400-speed-contrastive_xbd_sgd_normal"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
# config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xbd_sgd_normal_frozen_cls.py"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

#--------------------------------------------------

exp_name="tsm-k400-speed-contrastive_xd_sgd_speed"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_speed_frozen_cls.py"

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

#--------------------------------------------------

exp_name="tsm-k400-speed-contrastive_xd_sgd_normal"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V1"
config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/speed_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_normal_frozen_cls.py"

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic

#--------------------------------------------------