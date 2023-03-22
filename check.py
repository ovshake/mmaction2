# # import pickle

# # with open('/data/shinpaul14/projects/mmaction2/work_dirs/tsm_r50_1x1x3_100e_ekmmsada_rgb_color_contrastive_V2_cls/tsm-k400-color-contrastive_xd_sgd_color_temp_50/train_D1_test_D1/output_eval.pkl', 'rb') as f:
# # 	data = pickle.load(f)
# # print(data[9])
# import torch
# from torch import nn

# a = [[1,2,3],[2,3,4],[4,3,1]]
# x = torch.tensor(a)
# print(x.shape)
# print(x.exp())
# denominator=x.exp()
# diag_elems=torch.diagonal(denominator,0)
# print('diag_elems',diag_elems)
# denominator = denominator.sum(1)
# denominator = denominator - diag_elems
# print(denominator)
# # print(x.exp().sum()-s)

# loss = - torch.log(diag_elems / denominator).mean()
# print(loss)




# # 0 climb
# # 1 fencing
# # 2 golf
# # 3 kick_ball  
# # 4 pullup
# # 5 punch
# # 6 pushup
# # 7 ride_bike
# # 8 ride_horse
# # 9 shoot_ball 
# # 10 shoot_bow
# # 11 walk

class_name = {0:'climb', 1:'fencing', 2:'golf',3:'kick_ball',4:'pullup',5:'punch',6:'pushup', 7:'ride_bike',8:'ride_horse',9:'shoot_ball',10:'shoot_bow',11:'walk'}
class_name_num = {'climb':0, 'fencing':1, 'golf':2,'kick_ball':3,'pullup':4,'punch':5,'pushup':6, 'ride_bike':7,'ride_horse':8,'shoot_ball':9,'shoot_bow':10,'walk':11}
import os


f = open("/data/jongmin/projects/mmaction2_paul_work/hmdb_val_TranSVAE.txt", 'r')
#f = open("/data/jongmin/projects/mmaction2_paul_work/hmdb_train_TranSVAE.txt", 'r')

lines = f.read()
#path = '/local_datasets/hmdb_ucf/hmdb51/videos/'
path = '/local_datasets/hmdb51/rawframes/'
lines
list_vid_path = []
vid_class_num = []
vid = []
x = lines.split('\n')
for line in x:
    if line =='':
        pass
    else:
        q = line.split(' ')
        video_name = q[0]

        video_name = video_name.split('/')
        class_name_1 = q[-1]

        vid_name = video_name[-1]
        #video_path =vid_name+'.avi'+ ' '+class_name_1
        video_path =vid_name
        # video_path =vid_name

        # print(video_path)
            # print(q)


#         video_name = q[0]
#         video_name = video_name.split('/')[-1]
#         video_name = video_name+'.avi'
#         # print(type(q[-1]))

#         number= int(q[-1])
#         name = class_name[number]
        list_vid_path.append(video_path)
        vid_class_num.append(class_name_1)
print(len(list_vid_path))
print(len(vid_class_num))
for a in range(11):
    print(class_name[a],vid_class_num.count(str(a)))
# print((list_hmdb))
# print(list_hmdb)
# print(list_hmdb)
# print(os.listdir(path))
idx=0
count_class = []
used_class = []
# print(os.listdir(path))
# print(len(os.listdir(path)))
vid_dir_list=[]
for x in range(len(list_vid_path)):
    # idx += 1
	if os.path.exists(os.path.join(path, class_name[int(vid_class_num[x])], list_vid_path[x] )):
		used_class.append(int(vid_class_num[x]))
		idx +=1
	else:
		count_class.append(int(vid_class_num[x]))
print(idx)
print(len(count_class))
for i in range(11):
    print(class_name[i],count_class.count(i))
    # for classes in os.listdir(path):
    #     vid_dir_list += os.listdir(os.path.join(path,classes))
# # #         vid_path = classes.split(' ')
        #vid_name , class_numbers= list_vid_path[x], vid_class_num[x]
#         if vid_name in os.listdir(os.path.join(path,classes)):

#             video_path_check = os.path.join(path, classes,vid_name)
# # #         # video_path = os.path.join(path,vid_path[0])
# # #         # video_path = os.path.join(path,classes,name_vid)
#         #if os.path.exists(video_path_check):
#             #print(video_path_check)
#             # frame_num = len(os.listdir(video_path_check))
#             #video_path = video_path# + ' '+str(frame_num)+' '+ class_numbers
#             vid.append(video_path_check)

        # else:
        #     print(video_path)
# #             video_path = video_path + ' '+ str(class_name_num[classes])
# for x in list_vid_path:
#     if x in vid_dir_list:
#         idx +=1
#     else:
#         pass
# print(idx)           
# print(len(vid))
# print(vid)
# file = open('/data/shinpaul14/projects/mmaction2/data_list/val_hmdb.txt','w')
# for item in vid:
#     file.write(item+"\n")

# file.close()

# print(vid)

        




#     #video_path = path + name +'/'+video_name+'.avi'
#     video = video_name+'.avi'
#     list_hmdb.append(video)

# for z in os.listdir(path):
#     print(z)