
# exp_name="tsm-mmsada-projhead-distillation"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_distillation"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_projhead_distillation_rgb.py"

exp_name="color-contrastive-single-instance-X-b-d-mean-simsiam_proj_layer_1"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_cont_ablation"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_rgb_simsiam_xdb.py"


work_dir="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name"

d1_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_65.pth"


# d2_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_80.pth"


# d3_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_85.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'


# exp_name="tsm-mmsada-projhead-speed-distillation"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_distillation"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_projhead_distillation_rgb_speed.py"

# work_dir="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name"

# d1_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_100.pth"


# d2_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_90.pth"


# d3_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_55.pth"


# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'


# exp_name="tsm-mmsada-projhead-color-distillation"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_distillation"
# config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_projhead_distillation_rgb_color.py"

# work_dir="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name"

# d1_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_100.pth"


# d2_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_90.pth"


# d3_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_55.pth"


# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'