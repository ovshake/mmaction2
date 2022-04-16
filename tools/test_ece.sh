
section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v3"

exp_name="speed_color_augself_contrastive"


config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_k400_ekmmsada_multiple_contrastive_space_augself.py"


work_dir="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name"

d1_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D1_test_D1/epoch_85.pth"


d2_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D2_test_D2/epoch_80.pth"


d3_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D3_test_D3/epoch_95.pth"

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2' 

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D3' 

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D1' 

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy  --cfg-options data.test.domain='D2' 
