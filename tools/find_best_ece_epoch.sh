#!/bin/bash

section_name=$1
run_name=$2 
config_name=$3 

# domain=1


# for epoch in {5..100..5};  do 
#     ckpt_path="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$section_name/$run_name/train_D${domain}_test_D${domain}/epoch_${epoch}.pth"
#     echo $ckpt_path
#     python tools/test.py $config_name $ckpt_path --out work_dirs/test/output.pkl --eval ece_score   --cfg-options data.test.domain='D1'
# done


domain=2


for epoch in {5..100..5};  do 
    ckpt_path="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$section_name/$run_name/train_D${domain}_test_D${domain}/epoch_${epoch}.pth"
    echo $ckpt_path
    python tools/test.py $config_name $ckpt_path --out work_dirs/test/output.pkl --eval ece_score   --cfg-options data.test.domain='D2'
done


# domain=3


# for epoch in {5..100..5};  do 
#     ckpt_path="/data/jongmin/projects/mmaction2_paul_work/work_dirs/$section_name/$run_name/train_D${domain}_test_D${domain}/epoch_${epoch}.pth"
#     echo $ckpt_path
#     python tools/test.py $config_name $ckpt_path --out work_dirs/test/output.pkl --eval ece_score   --cfg-options data.test.domain='D3'
# done

