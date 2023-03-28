


import json
import os

color_dir = '/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_color_pathway_w_pretrained_linear/'
speed_dir = '/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/tsm_k400_normal_2_speed_pathway_w_pretrained_avg_linear/'
vcop_dir = '/data/shinpaul14/projects/mmaction2/work_dirs/Teacher_model/VCOP_4_clip/'
test_domain = ['train_D1_test_D1', 'train_D1_test_D2','train_D1_test_D3','train_D2_test_D1', 
               'train_D2_test_D2','train_D2_test_D3','train_D3_test_D1', 'train_D3_test_D2','train_D3_test_D3'  ]
for x in test_domain:
# Open the JSON file and load the data
    color_json_path = color_dir+ x + '/prediction.json'
    speed_json_path = speed_dir+ x + '/prediction.json'
    vcop_json_path = vcop_dir+ x + '/prediction.json'
    with open(color_json_path, "r") as f:
        color = json.load(f)
    with open(speed_json_path, "r") as q:
        speed = json.load(q)
    with open(vcop_json_path, "r") as w:
        vcop = json.load(w)

    # Access the values associated with the labels and pred keys
    gt = color["label"]
    pred_1 = color["pred"]
    pred_2 = speed["pred"]
    pred_3 = vcop["pred"]

    model_num=2


    num_correct = 0
    num_total = 0

    for idx, ground_truth in enumerate(gt):
        if model_num == 3:
            if ground_truth == pred_1[idx] and ground_truth == pred_2[idx] and ground_truth == pred_3[idx]:
                num_correct += 1
            num_total += 1
        elif model_num == 2:
            if ground_truth == pred_1[idx] and ground_truth == pred_2[idx]:
                num_correct += 1
            num_total += 1

    accuracy = round((num_correct / num_total)*100, 3)

    print("Accuracy {}:".format(x), accuracy)