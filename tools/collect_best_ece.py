import os 
import glob 

def get_best_ece_from_run_path(json_log):
    with open(json_log, 'r') as f:
        lines = f.readlines() 
    
    best_ece = float('inf')
    val_acc = -1
    epoch = 0
    for dict_str in lines:
        dict_str = dict_str.rstrip() 
        dict_str = dict_str.replace("null", 'None')
        epoch_dict = eval(dict_str) 
        
        if "mode" in epoch_dict and epoch_dict["mode"] == "val":
            if epoch_dict["ece"] < best_ece:
                best_ece = epoch_dict["ece"]
                epoch = epoch_dict["epoch"]
                val_acc = epoch_dict["top1_acc"]

    
    return {'ece': best_ece, 'epoch': epoch, 'val_acc': val_acc} 

if __name__ == '__main__':
    run_folder_pattern = "/data/abhishek/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_v3/**/**/*.log.json"
    folder_paths = glob.glob(run_folder_pattern, recursive=True) 
    for path in folder_paths:
        run_name = path.split('/')[-2] 
        model_name = path.split('/')[-3]
        best_val = get_best_ece_from_run_path(path) 
        print(f"Model Name: {model_name} Run Name: {path} - Best Ece: {best_val}")
