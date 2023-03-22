exp_name="tsm-k400-speed-contrastive_xd_sgd_speed_temp_5_1"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_speed_contrastive_V2_cls"

config="/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_rgb.py"


work_dir="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name"

d1_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_80.pth"

d2_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_55.pth"

d3_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_60.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'



#----------------------------



# exp_name="tsm-k400-color-simsiam_sp_pathway_B_color_batch40"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_simsiam"
# config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_rgb_simsiam_xdb.py'



# work_dir="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name"

# d1_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_100.pth"


# d2_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_55.pth"


# d3_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_85.pth"


# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'



# #----------------------------



# exp_name="tsm-k400-color-simsiam_sp_pathway_A_normal_batch40"
# exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_simsiam"
# config='/data/jongmin/projects/mmaction2_paul_work/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_contrastivehead_ekmmsada_rgb_simsiam_xdb_sp_normal.py'



# work_dir="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name"

# d1_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_75.pth"


# d2_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_35.pth"


# d3_ckpt="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_85.pth"


# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'