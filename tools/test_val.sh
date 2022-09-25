
exp_name="speed_plus_color_contrastive_xdb_pw-2-sgp_no_proj_head"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_v1"
config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ekmmsada_multiple_contrastive_space.py"

work_dir="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name"

d1_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_45.pth"


d2_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_90.pth"


d3_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_20.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'
