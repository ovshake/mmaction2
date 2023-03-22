


exp_name="tsm-k400-color-simsiam_distill_speed_contrastive_head"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_simsiam_distill"
config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_distillation_rgb_speed.py"


# exp_name="trash"
# exp_section="trash"
# condif='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_rgb.py'

work_dir="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name"
#d1_ckpt = '/data/jongmin/projects/mmaction2_paul_work/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_simsiam_distill/tsm-k400-color-simsiam_distill_color/train_D1_test_D1/best_top1_acc_epoch_100.pth'
d1_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_10.pth"





python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'
