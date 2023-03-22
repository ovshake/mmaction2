#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J simsiam_xdb_sgd_color_TSMconstructor
#SBATCH -o /data/jongmin/outputs/simsiam_xdb_sgd_color_TSMconstructor.out
#SBATCH -e /data/jongmin/outputs/simsiam_xdb_sgd_color_TSMconstructor.err
#SBATCH --time 1-0
#SBATCH -p batch_grad
#SBATCH -x ai[5,10]
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 64G


. /data/jongmin/anaconda3/etc/profile.d/conda.sh
conda activate action-dg




exp_name="tsm-k400-color-simsiam_xdb_sgd_color_TSMconstructor"
#exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_simsiam_V1"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_simsiam_scratch"
config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/train_scratch/tsm_r50_1x1x3_100e_colorspatial_ekmmsada_rgb_real_simsiam_xdb_sgd_color_TSMconstructor.py"

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=100  --validate --deterministic