#!/bin/bash 
config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_augself_contrastivehead_ekmmsada_rgb.py"

d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v2/color-jitter-augself-contrastive-10x/train_D1_test_D1/best_top1_acc_epoch_90.pth"
d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v2/color-jitter-augself-contrastive-10x/train_D2_test_D2/best_top1_acc_epoch_45.pth"
d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v2/color-jitter-augself-contrastive-10x/train_D3_test_D3/best_top1_acc_epoch_45.pth"


echo "$config"
echo "D1 -> D1"
python tools/test.py $config $d1_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D1'

echo "D1 -> D2"
python tools/test.py $config $d1_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D2'

echo "D1 -> D3"
python tools/test.py $config $d1_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D3'

echo "D2 -> D1"
python tools/test.py $config $d2_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D1'

echo "D2 -> D2"
python tools/test.py $config $d2_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D2'

echo "D2 -> D3"
python tools/test.py $config $d2_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D3'

echo "D3 -> D1"
python tools/test.py $config $d3_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D1'

echo "D3 -> D2"
python tools/test.py $config $d3_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D2'
echo "D3 -> D3"
python tools/test.py $config $d3_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D3'



config="/data/shinpaul14/projects/mmaction2/configs/recognition/tsm/tsm_r50_1x1x3_100e_colorspatial_augself_ekmmsada_rgb.py"

d1_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v2/colorjitter-augself-10x/train_D1_test_D1/best_top1_acc_epoch_50.pth"
d2_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v2/colorjitter-augself-10x/train_D2_test_D2/best_top1_acc_epoch_45.pth"
d3_ckpt="/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v2/colorjitter-augself-10x/train_D3_test_D3/best_top1_acc_epoch_50.pth"



echo "$config"

echo "D1 -> D1"
python tools/test.py $config $d1_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D1'

echo "D1 -> D2"
python tools/test.py $config $d1_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D2'

echo "D1 -> D3"
python tools/test.py $config $d1_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D3'

echo "D2 -> D1"
python tools/test.py $config $d2_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D1'

echo "D2 -> D2"
python tools/test.py $config $d2_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D2'
echo "D2 -> D3"
python tools/test.py $config $d2_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D3'

echo "D3 -> D1"
python tools/test.py $config $d3_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D1'

echo "D3 -> D2"
python tools/test.py $config $d3_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D2'
echo "D3 -> D3"
python tools/test.py $config $d3_ckpt --out work_dirs/test/output.pkl   --eval ece_score   --cfg-options data.test.domain='D3'
