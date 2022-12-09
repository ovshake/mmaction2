

exp_name="trash_frozen"
exp_section="trash_frozen"

# config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/multi_contrastive/tsm_r50_1x1x3_100e_multi_contrastive_ekmmsada_rgb_simsiam_xadb_sgd.py'


# #bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=1 --validate --deterministic 
# python /data/shinpaul14/projects/mmaction2/tools/train.py $config --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=1 --validate --deterministic 


config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/color_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xbd_sgd_color_frozen_cls.py"
#config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/color_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xbd_sgd_color_frozen_cls.py'

# exp_name1="tsm-k400-color-contrastive_xd_sgd_color"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py"
# exp_name="tsm-k400-color-contrastive_xd_sgd_color"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2"

#bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=2  --validate --deterministic
python /data/shinpaul14/projects/mmaction2/tools/train.py $config --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' total_epochs=3 --validate --deterministic 
#PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/best_top1_acc_epoch_60.pth" total_epochs=1  --validate --deterministic


# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2 data.train.domain='D2' data.val.domain='D2' load_from="./work_dirs/$exp_section1/$exp_name1/train_D2_test_D2/best_top1_acc_epoch_55.pth" total_epochs=100  --validate --deterministic

# PORT=7070 bash /data/shinpaul14/projects/mmaction2/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3 data.train.domain='D3' data.val.domain='D3' load_from="./work_dirs/$exp_section1/$exp_name1/train_D3_test_D3/best_top1_acc_epoch_85.pth" total_epochs=100  --validate --deterministic