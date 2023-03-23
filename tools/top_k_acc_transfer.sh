# !/bin/bash

exp_name="order-contrastive"
section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2"
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_order_contrastivehead_ekmmsada_rgb.py"
work_dir="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name"
d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D1_test_D1/epoch_100.pth"
d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D2_test_D2/epoch_80.pth"
d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D3_test_D3/epoch_85.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D1/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1' 

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2' 

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D2/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3' 


python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1' 

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2' 

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D3/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3' 



# exp_name="tsm-baseline-mmsada"
# section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2"
# config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_rgb.py"
# work_dir="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name"
# d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D1_test_D1/epoch_100.pth"
# d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D2_test_D2/epoch_85.pth"
# d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D3_test_D3/epoch_95.pth"


# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2' 

# python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3' 

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1' 

# python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3' 

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1' 

# python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/top_k_acc_on_best_ece_output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2' 

