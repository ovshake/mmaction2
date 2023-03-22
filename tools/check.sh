

exp_name="trash_frozen"
exp_section="trash_frozen"

# config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/color_symmetric_contrstive/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_color_frozen_cls.py'
# exp_name1="tsm-k400-color-contrastive_xd_sgd_color"
# exp_section1="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"
#config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/color_symmetric_contrstive/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_fast_simsiam_frozen_cls_batch.py"
#config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/color_symmetric_contrstive/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_color_frozen_cls.py'

config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/distillation/tsm_r50_1x1x3_100e_ekmmsada_distillation_rgb.py"
#config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/late_fusion/tsm_r50_1x1x3_100e_latefusion_speed_color_vcop_input_normal_cls.py'

#config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb.py'

#config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_resfrozen_rgb_decay_0.py'
# exp_name="tsm-k400-color-contrastive_xd_sgd_color_temp_5"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_cls"


#python /data/jongmin/projects/mmaction2_paul_work/tools/train.py $config  --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D1' data.val.domain='D1' load_from="./work_dirs/$exp_section1/$exp_name1/train_D1_test_D1/latest.pth" total_epochs=1  --validate --deterministic
bash /data/jongmin/projects/mmaction2_paul_work/tools/dist_train.sh $config 4 --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D3' data.val.domain='D3' total_epochs=2 --validate --deterministic 
# python /data/jongmin/projects/mmaction2_paul_work/tools/train.py $config --cfg-options work_dir=/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1 data.train.domain='D3' data.val.domain='D3' total_epochs=3 --validate --deterministic 
