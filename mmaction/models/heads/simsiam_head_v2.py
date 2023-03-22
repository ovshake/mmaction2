# import torch
# import torch.nn as nn
# from mmcv.cnn import normal_init
# import torch.nn.functional as F
# from einops import rearrange

# from ..builder import HEADS
# from .base import AvgConsensus, BaseHead
# from ..builder import build_loss
# import math
# import numpy as np
# import itertools









# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# @HEADS.register_module()
# class prediction_MLP(nn.Module):
#     def __init__(self,
#                  feature_size,
#                  init_std=0.001,
#                  middle_layer_dim=512,
#                  img_dim=2048,
#                  **kwargs):

#         super().__init__()


#         self.fc1 = nn.Linear(feature_size, middle_layer_dim, bias=False)
#         self.BN1 = nn.BatchNorm1d(middle_layer_dim)
#         self.relu_1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(middle_layer_dim, img_dim)



#         self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2)

#         self.init_std = init_std
#         self.img_dim = img_dim
#         self.init_weights()



#     def forward(self, features):
#         #features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
#         features = self.encoder(features)
#         return features

#     def init_weights(self):
#         """Initiate the parameters from scratch."""
#         normal_init(self.encoder, std=self.init_std)



# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



# @HEADS.register_module()
# class projection_MLP(nn.Module):
#     def __init__(self,
#                  feature_size,
#                  num_segments,
#                  init_std=0.001,
#                  middle_layer_dim=2048,
#                  img_dim=2048,
#                  **kwargs):

#         super().__init__()


#         self.fc1 = nn.Linear(feature_size * num_segments , middle_layer_dim, bias=False)
#         self.BN1 = nn.BatchNorm1d(middle_layer_dim)
#         self.relu_1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=False)
#         self.BN2 = nn.BatchNorm1d(img_dim)
#         self.relu_2 = nn.ReLU(inplace=True)
#         self.fc3 = nn.Linear(img_dim, img_dim, bias=False)
#         self.BN3 = nn.BatchNorm1d(img_dim, affine=False)

#         self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2, self.BN2, self.relu_2, self.fc3, self.BN3)

#         self.init_std = init_std
#         self.num_segments = num_segments
#         self.img_dim = img_dim
#         self.init_weights()



#     def forward(self, features):
#         # this reduces the dimensionality of the features to the batch size where single video of N frames are concat on dim=1 and through the FC layer it is reduced to 2048
#         features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
   
#         features = self.encoder(features)
#         return features

#     def init_weights(self):
#         """Initiate the parameters from scratch."""
#         normal_init(self.encoder, std=self.init_std)




# #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# # projection that averages the N frames of features 


# @HEADS.register_module()
# class projection_MLP_avg(nn.Module):
#     def __init__(self,
#                  feature_size,
#                  num_segments,
#                  init_std=0.001,
#                  middle_layer_dim=2048,
#                  img_dim=2048,
#                  spatial_type='avg',
#                  consensus=dict(type='AvgConsensus', dim=1),
#                  **kwargs):

#         super().__init__()




#         self.fc1 = nn.Linear(feature_size * num_segments , middle_layer_dim, bias=False)
#         self.BN1 = nn.BatchNorm1d(middle_layer_dim)
#         self.relu_1 = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(middle_layer_dim, img_dim, bias=False)
#         self.BN2 = nn.BatchNorm1d(img_dim)
#         self.relu_2 = nn.ReLU(inplace=True)
#         self.fc3 = nn.Linear(img_dim, img_dim, bias=False)
#         self.BN3 = nn.BatchNorm1d(img_dim, affine=False)

#         self.encoder = nn.Sequential(self.fc1, self.BN1, self.relu_1, self.fc2, self.BN2, self.relu_2, self.fc3, self.BN3)

#         self.init_std = init_std
#         self.num_segments = num_segments
#         self.img_dim = img_dim

#         self.spatial_type = spatial_type
#         consensus_ = consensus.copy()

#         consensus_type = consensus_.pop('type')
#         if consensus_type == 'AvgConsensus':
#             self.consensus = AvgConsensus(**consensus_)
#         else:
#             self.consensus = None
        
#         if self.spatial_type == 'avg':
#             # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
#             self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         else:
#             self.avg_pool = None



#         self.init_weights()



#     def forward(self, features):
#         #[N x Batch, 2048]
#         # average N frames of features
#         if self.spatial_type == 'avg':
#             features = self.avg_pool(features)
#         elif self.spatial_type == 'frame':
#             features = features
#         else:
#         # this reduces the dimensionality of the features to the batch size where single video of N frames are concat on dim=1 and through the FC layer it is reduced to 2048
#             features = rearrange(features, '(b c) e -> b (c e)', c=self.num_segments)
   
#         features = self.encoder(features)
#         return features

#     def init_weights(self):
#         """Initiate the parameters from scratch."""
#         normal_init(self.encoder, std=self.init_std)