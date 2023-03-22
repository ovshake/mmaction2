#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH -J multi_contrastive_xabd_keys_sgd_cls
#SBATCH -o /data/jongmin/outputs/multi_contrastive_xabd_keys_sgd_cls.out
#SBATCH -e /data/jongmin/outputs/multi_contrastive_xabd_keys_sgd_cls.err
#SBATCH --time 2-0
#SBATCH -p batch_grad
#SBATCH -x ai[10]
#SBATCH --cpus-per-gpu 4
#SBATCH --mem 64G


. /data/jongmin/anaconda3/etc/profile.d/conda.sh
conda activate action-dg


exp_name1="tsm-k400-multi-contrastive-xabd-keys-sgd_simsiam"
exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v1"
config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'



exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_simsiam_best200_cls"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v2"

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_100.pth" total_epochs=200  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_45.pth" total_epochs=200  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_55.pth" total_epochs=200  --validate --deterministic


exp_name1="tsm-k400-multi-contrastive-xabd-keys-sgd_looc"
exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v1"
config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_looc_best_200cls"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v2"
PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_60.pth" total_epochs=100  --validate --deterministic

PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_80.pth" total_epochs=100  --validate --deterministic





# exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_simsiam_3heads"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v1"
# config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
# # exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_simsiam_3heads_last"
# # exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v2"

# # PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

# # PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

# # PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic

# exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_simsiam_3heads_best"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v2"
# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_15.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_90.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_45.pth" total_epochs=100  --validate --deterministic


# exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_3head"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v1"
# config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'
# # exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_3head_last"
# # exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v2"

# # PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=100  --validate --deterministic

# # PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/latest.pth" total_epochs=100  --validate --deterministic

# # PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/latest.pth" total_epochs=100  --validate --deterministic

# exp_name="tsm-k400-multi-contrastive-xabd-keys-sgd_3head_best"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_multi_contrastive_v2"

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_60.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_80.pth" total_epochs=100  --validate --deterministic




