exp_name="tsm-k400-color-normal-simsiam"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_rgb_color_simsiam_V2_cls"
#config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/late_fusion/tsm_r50_1x1x3_100e_latefusion_vcop_speed_input_normal_cls.py'
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_rgb.py"


work_dir="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name"

d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_25.pth"

d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_10.pth"

d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_40.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'


