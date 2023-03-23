exp_name="tsm_k400_CPS_CS_vcop"
exp_section="ICMR_late_fusion"
config='/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/ICMR_ensemble/tsm_r50_1x1x3_100e_latefusion_CPS_CS_concat_vcop.py'

#//data/shinpaul14/projects/mmaction2/work_dirs/ICMR_distil_final_new_cls/tsm_k400_distillation_all_each

# exp_name="tsm_k400_CS_NC_vcop"
# exp_section="ICMR_late_fusion"
# config='/data/shinpaul14/projects/mmaction2/work_dirs/ICMR_late_fusion/tsm_k400_CS_NC_vcop/train_D1_test_D1/tsm_r50_1x1x3_100e_latefusion_NC_CS_concat_vcop.py'

work_dir="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name"

d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D1_test_D1/best_top1_acc_epoch_80.pth"

d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D2_test_D2/best_top1_acc_epoch_100.pth"

d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$exp_section/$exp_name/train_D3_test_D3/best_top1_acc_epoch_45.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D1' data.val.domain='D1' model.domain='D1' data.test.domain='D1'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D1' data.val.domain='D1' model.domain='D1' data.test.domain='D2'

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D1' data.val.domain='D1' model.domain='D1' data.test.domain='D3'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D2' data.val.domain='D2' model.domain='D2' data.test.domain='D1'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D2' data.val.domain='D2' model.domain='D2' data.test.domain='D2'

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D2' data.val.domain='D2' model.domain='D2' data.test.domain='D3'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D3' data.val.domain='D3' model.domain='D3' data.test.domain='D1'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D3' data.val.domain='D3' model.domain='D3' data.test.domain='D2'

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/output_eval.pkl" --eval top_k_accuracy  --cfg-options data.train.domain='D3' data.val.domain='D3' model.domain='D3' data.test.domain='D3'


# /data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_simsiam_5_V2_cls/tsm-k400-both-normal-speed-simsiam