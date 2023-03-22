exp_name="tsm-k400-color-contrastive_xd_sgd_color_temp_50"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V1"

#config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_rgb_recall.py"

config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/color_contrastive_stage1/tsm_r50_1x1x3_k400_100e_colorspatial_ekmmsada_rgb_contrastive_xd_sgd_color_frozen_cls_temp_5_recall.py'
work_dir="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name"

#d1_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_80.pth"
d1_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1/latest.pth"



python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_rgb_recall.py"
#python tools/compute_recall_at_k.py --annotation-file "/data/jongmin/projects/SADA_Domain_Adaptation_Splits/D1_test.pkl" --output_file "/data/jongmin/projects/mmaction2_paul_work/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_cls/tsm-k400-color-contrastive_xd_sgd_color_temp_50/train_D1_test_D1/output_eval.pkl"