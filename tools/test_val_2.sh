exp_name="tsm_k400_supcon_normal2color_vesion_no_affine_dg"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_reproduce_cls"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_rgb.py'
#//data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/VCOP_4_clip

work_dir="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name"


d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_*.pth"


d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_*.pth"


d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_*.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy_all_in_one   --cfg-options data.test.domain='D1'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D2'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy_all_in_one  --cfg-options data.test.domain='D3'

