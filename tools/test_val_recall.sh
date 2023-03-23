exp_name="tsm-k400-color_p_speed_color_pathway"
exp_section="tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V1"

# /data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_temp_5_batch_12_V2_cls/tsm-k400-color_p_speed_color_pathway
#config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_baseline_batch_12.py'
#condif='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_baseline_batch_16.py'
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_baseline/tsm_r50_1x1x3_100e_ekmmsada_baseline_batch_12_extract_feature.py' 


work_dir="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name"


d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_90.pth"

d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_55.pth"

d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_60.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1'

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval_back.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3'


# /data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_simsiam_5_V2_cls/tsm-k400-both-normal-speed-simsiam