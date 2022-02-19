section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2"

exp_name="tsm-baseline-mmsada"


config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_ekmmsada_rgb.py"


work_dir="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name"

d1_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D1_test_D1/best_top1_acc_epoch_50.pth"


d2_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D2_test_D2/best_top1_acc_epoch_50.pth"

d3_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D3_test_D3/best_top1_acc_epoch_25.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D2' 

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D3' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D1' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D3' 


python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D1' 

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D2' 









section_name="tsm_r50_1x1x3_100e_ekmmsada_rgb_v2"

exp_name="color-jitter-moco-m-99-q-4096"


config="/data/abhishek/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_moco_contrastivehead_ekmmsada_rgb.py"


work_dir="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name"

d1_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D1_test_D1/best_top1_acc_epoch_80.pth"


d2_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D2_test_D2/best_top1_acc_epoch_55.pth"

d3_ckpt="/data/abhishek/projects/mmaction2/work_dirs/$section_name/$exp_name/train_D3_test_D3/best_top1_acc_epoch_10.pth"


python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D2/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D2' 

python tools/test.py $config $d1_ckpt --out "$work_dir/train_D1_test_D3/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D3' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D1/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D1' 

python tools/test.py $config $d2_ckpt --out "$work_dir/train_D2_test_D3/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D3' 


python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D1/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D1' 

python tools/test.py $config $d3_ckpt --out "$work_dir/train_D3_test_D2/output.pkl" --eval top_k_accuracy --cfg-options data.test.domain='D2' 