#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J multi_contrastive_xabd_keys_sgd_simsiam
#SBATCH -o /data/jongmin/outputs/multi_contrastive_xabd_keys_sgd.out
#SBATCH -e /data/jongmin/outputs/multi_contrastive_xabd_keys_sgd.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[10]
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 100G


. /data/jongmin/anaconda3/etc/profile.d/conda.sh
conda activate action-dg







exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_simsiam_200"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v1"
config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/multi_contrastive/tsm_r50_1x1x3_100e_multi_contrastive_ekmmsada_rgb_simsiam_xadb_sgd.py"

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=200  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' total_epochs=200  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' total_epochs=200  --validate --deterministic

